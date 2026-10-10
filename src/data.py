"""Synthetic terrestrial-laser-scan scenes for point-cloud semantic segmentation.

Each scene imitates a single static scan from a tripod scanner (like a Leica RTC360/BLK
style instrument) standing at the origin, 1.8 m above the ground:

* surfaces are sampled for ground, building facades, poles, trees and vehicles,
* point density falls off with range (roughly 1/r^2, like a real scanner),
* range noise grows with distance,
* "noise" points (dust, rain, multipath, stray returns) float in free space,
* every point carries an intensity value that depends on material and range.

The goal is not photo-realism; it is to give you data with the same *shape* problems as
real scans (class imbalance, sparse far points, outliers, mixed feature ranges) so that
training, evaluation and quantization behave in a realistic way, and everything runs on a
laptop in minutes without downloading a dataset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["ground", "building", "pole", "vegetation", "vehicle", "noise"]
NUM_CLASSES = len(CLASS_NAMES)
# RGB colours used by the notebooks for plots.
CLASS_COLORS = np.array(
    [
        [0.55, 0.45, 0.35],  # ground
        [0.85, 0.30, 0.25],  # building
        [0.20, 0.20, 0.20],  # pole
        [0.25, 0.65, 0.25],  # vegetation
        [0.20, 0.45, 0.85],  # vehicle
        [0.95, 0.75, 0.10],  # noise
    ]
)

SCANNER_HEIGHT = 1.8  # metres above ground
SCENE_RADIUS = 25.0  # metres; also used to normalize coordinates for the model
_BASE_INTENSITY = np.array([0.35, 0.55, 0.70, 0.25, 0.80, 0.10])

SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


# ---------------------------------------------------------------------------
# Primitive surface samplers (all return (M, 3) arrays in metres)
# ---------------------------------------------------------------------------

def _ground(rng, n):
    r = SCENE_RADIUS * np.sqrt(rng.random(n))
    phi = rng.random(n) * 2 * np.pi
    x, y = r * np.cos(phi), r * np.sin(phi)
    slope = rng.normal(0, 0.02, size=2)
    z = slope[0] * x + slope[1] * y + 0.08 * np.sin(0.4 * x) * np.cos(0.3 * y)
    return np.stack([x, y, z], 1)


def _ground_height(x, y):
    # Objects are placed on z=0; the gentle ground undulation is ignored for them.
    return np.zeros_like(x)


def _rotate_z(p, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    out = p.copy()
    out[:, 0] = c * p[:, 0] - s * p[:, 1]
    out[:, 1] = s * p[:, 0] + c * p[:, 1]
    return out


def _box_surface(rng, n, size, include_top=True, include_bottom=False):
    """Uniform samples on the surface of an axis-aligned box centred at the origin (z from 0)."""
    lx, ly, lz = size
    faces = [
        ("x", lx / 2, ly * lz), ("x", -lx / 2, ly * lz),
        ("y", ly / 2, lx * lz), ("y", -ly / 2, lx * lz),
    ]
    if include_top:
        faces.append(("z", lz, lx * ly))
    if include_bottom:
        faces.append(("z", 0.0, lx * ly))
    areas = np.array([f[2] for f in faces])
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for (axis, val, _), k in zip(faces, counts):
        u = rng.random((k, 3)) * np.array([lx, ly, lz]) - np.array([lx / 2, ly / 2, 0])
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        u[:, idx] = val
        pts.append(u)
    return np.concatenate(pts, 0)


def _cylinder_surface(rng, n, radius, height):
    phi = rng.random(n) * 2 * np.pi
    z = rng.random(n) * height
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], 1)


def _ellipsoid_shell(rng, n, radii, thickness=0.35):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    scale = 1.0 - thickness * rng.random(n)  # points fill an outer shell: leafy canopy
    return v * scale[:, None] * np.asarray(radii)


def _random_position(rng, r_min, r_max):
    r = rng.uniform(r_min, r_max)
    phi = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


# ---------------------------------------------------------------------------
# Scene generator
# ---------------------------------------------------------------------------

def make_scene(rng: np.random.Generator, n_points: int = 2048, oversample: int = 6):
    """Generate one scan.

    Returns
    -------
    xyz : (n_points, 3) float32, metres, scanner at (0, 0, SCANNER_HEIGHT)
    intensity : (n_points,) float32 in [0, 1]
    labels : (n_points,) int64 in [0, NUM_CLASSES)
    """
    budget = n_points * oversample
    pts, lab = [], []

    def add(p, cls):
        pts.append(p)
        lab.append(np.full(len(p), cls, dtype=np.int64))

    # Ground
    add(_ground(rng, int(budget * 0.35)), 0)

    # Buildings: big boxes further away
    for _ in range(rng.integers(1, 4)):
        size = (rng.uniform(6, 14), rng.uniform(6, 12), rng.uniform(4, 12))
        p = _box_surface(rng, int(budget * 0.12), size)
        p = _rotate_z(p, rng.uniform(0, np.pi))
        p[:, :2] += _random_position(rng, 13, 23)
        add(p, 1)

    # Poles: thin tall cylinders (lamp posts, sign posts)
    for _ in range(rng.integers(2, 7)):
        p = _cylinder_surface(rng, int(budget * 0.012), rng.uniform(0.08, 0.2), rng.uniform(4, 9))
        p[:, :2] += _random_position(rng, 3, 18)
        add(p, 2)

    # Trees: trunk + canopy, both "vegetation"
    for _ in range(rng.integers(2, 8)):
        trunk_h = rng.uniform(1.5, 3.0)
        canopy_r = rng.uniform(1.5, 3.0)
        base = _random_position(rng, 3, 22)
        trunk = _cylinder_surface(rng, int(budget * 0.006), rng.uniform(0.12, 0.3), trunk_h)
        canopy = _ellipsoid_shell(rng, int(budget * 0.03), (canopy_r, canopy_r, canopy_r * rng.uniform(0.8, 1.4)))
        canopy[:, 2] += trunk_h + canopy_r * 0.8
        tree = np.concatenate([trunk, canopy], 0)
        tree[:, :2] += base
        add(tree, 3)

    # Vehicles: car-sized boxes near the scanner
    for _ in range(rng.integers(1, 6)):
        size = (rng.uniform(4.2, 5.0), rng.uniform(1.7, 2.0), rng.uniform(1.4, 1.7))
        p = _box_surface(rng, int(budget * 0.025), size)
        p = _rotate_z(p, rng.uniform(0, np.pi))
        p[:, :2] += _random_position(rng, 3, 14)
        add(p, 4)

    xyz = np.concatenate(pts, 0)
    labels = np.concatenate(lab, 0)

    # --- Scanner physics: density falls with range, range noise grows with range ---
    scanner = np.array([0.0, 0.0, SCANNER_HEIGHT])
    rel = xyz - scanner
    rng_dist = np.linalg.norm(rel, axis=1)
    keep_prob = np.clip((4.0 / np.maximum(rng_dist, 1e-3)) ** 2, 0.04, 1.0)
    keep = rng.random(len(xyz)) < keep_prob
    xyz, labels, rel, rng_dist = xyz[keep], labels[keep], rel[keep], rng_dist[keep]
    ray = rel / rng_dist[:, None]
    xyz = xyz + ray * rng.normal(0, 0.005 + 0.0015 * rng_dist)[:, None]

    # --- Noise / outlier points: dust near the scanner and stray returns in free space ---
    n_noise = int(n_points * rng.uniform(0.01, 0.04))
    n_dust = n_noise // 3
    dust = scanner + rng.normal(0, 0.8, size=(n_dust, 3))
    stray = np.stack(
        [rng.uniform(-15, 15, n_noise - n_dust), rng.uniform(-15, 15, n_noise - n_dust), rng.uniform(0.5, 10, n_noise - n_dust)], 1
    )
    xyz = np.concatenate([xyz, dust, stray], 0)
    labels = np.concatenate([labels, np.full(n_noise, 5, dtype=np.int64)], 0)

    # --- Intensity: material reflectivity, attenuated with range ---
    dist = np.linalg.norm(xyz - scanner, axis=1)
    intensity = _BASE_INTENSITY[labels] + rng.normal(0, 0.08, len(labels))
    intensity = intensity * np.exp(-dist / 60.0)
    intensity = np.clip(intensity, 0.0, 1.0)

    # --- Fixed point count (static shapes for the model and the accelerator) ---
    replace = len(xyz) < n_points
    idx = rng.choice(len(xyz), n_points, replace=replace)
    return xyz[idx].astype(np.float32), intensity[idx].astype(np.float32), labels[idx]


def to_model_input(xyz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """(N,3)+(N,) -> (4, N) float32: x/R, y/R, z/R, intensity. Channels-first for Conv1d.

    Normalizing xyz to roughly [-1, 1] keeps every input channel in a similar range. That
    helps training, and it matters a lot for per-tensor input quantization (see notebook 02).
    """
    feats = np.concatenate([xyz / SCENE_RADIUS, intensity[:, None]], axis=1)
    return feats.T.astype(np.float32).copy()


def augment(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random yaw rotation, mild scaling and jitter (scanner stays at the origin)."""
    out = _rotate_z(xyz, rng.uniform(0, 2 * np.pi))
    out = out * rng.uniform(0.9, 1.1)
    out = out + rng.normal(0, 0.01, size=out.shape)
    return out.astype(np.float32)


class SyntheticScans(Dataset):
    """Deterministic dataset: scene i of a split is always the same (seeded by split + i)."""

    def __init__(self, split: str = "train", size: int = 2000, n_points: int = 2048, augment_data: bool | None = None):
        assert split in SPLIT_SEED_OFFSET, split
        self.split, self.size, self.n_points = split, size, n_points
        self.augment = (split == "train") if augment_data is None else augment_data
        self._aug_rng = np.random.default_rng()

    def __len__(self):
        return self.size

    def scene(self, i: int):
        rng = np.random.default_rng(SPLIT_SEED_OFFSET[self.split] + i)
        return make_scene(rng, self.n_points)

    def __getitem__(self, i):
        xyz, intensity, labels = self.scene(i)
        if self.augment:
            xyz = augment(xyz, self._aug_rng)
        x = to_model_input(xyz, intensity)
        return torch.from_numpy(x), torch.from_numpy(labels)


def class_frequencies(n_scenes: int = 100, n_points: int = 2048) -> np.ndarray:
    ds = SyntheticScans("train", size=n_scenes, n_points=n_points, augment_data=False)
    counts = np.zeros(NUM_CLASSES)
    for i in range(n_scenes):
        counts += np.bincount(ds.scene(i)[2], minlength=NUM_CLASSES)
    return counts / counts.sum()
