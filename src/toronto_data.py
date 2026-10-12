"""Toronto_3D PLY loader that turns large scans into fixed-size training tiles.

The Toronto scans are binary little-endian PLY files with per-point XYZ, RGB,
intensity, GPS time, scan angle and semantic label fields. This module reads the
vertex records through a memory map and caches one fixed-size sample per XY tile.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TORONTO_CLASS_NAMES = [
    "Unclassified",
    "Ground",
    "Road_markings",
    "Natural",
    "Building",
    "Utility_line",
    "Pole",
    "Car",
    "Fence",
]
NUM_TORONTO_CLASSES = len(TORONTO_CLASS_NAMES)
_CACHE_VERSION = 4

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def _read_header(path: Path) -> tuple[list[str], int]:
    lines = []
    with path.open("rb") as stream:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            line = raw_line.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        offset = stream.tell()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Expected binary little-endian PLY 1.0: {path}")
    return lines, offset


def _vertex_layout(header: list[str]) -> tuple[int, np.dtype]:
    vertex_count = None
    fields = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts and parts[0] == "property":
            if len(parts) != 3 or parts[1] not in _PLY_TYPES:
                raise ValueError(f"Unsupported PLY vertex property: {line}")
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    if vertex_count is None or not fields:
        raise ValueError("PLY header has no vertex element or vertex properties")
    return vertex_count, np.dtype(fields, align=False)


def _cache_paths(scan: Path, cache_dir: Path, tile_size: float, n_points: int,
                 min_points: int, coord_scale: float) -> tuple[Path, Path, Path]:
    stat = scan.stat()
    header, _ = _read_header(scan)
    cache_key = f"{_CACHE_VERSION}:{scan.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{tile_size}:{n_points}:{min_points}:{coord_scale}:{'|'.join(header)}"
    digest = hashlib.sha1(cache_key.encode()).hexdigest()[:12]
    stem = f"{scan.stem}_{digest}"
    return cache_dir / f"{stem}_x.npy", cache_dir / f"{stem}_y.npy", cache_dir / f"{stem}.json"


def _build_scan_cache(scan: Path, x_path: Path, y_path: Path, meta_path: Path,
                      tile_size: float, n_points: int, min_points: int,
                      coord_scale: float, seed: int) -> None:
    """Build fixed-size per-tile training samples; temporary indexing is disk-backed."""
    header, payload_offset = _read_header(scan)
    vertex_count, dtype = _vertex_layout(header)
    names = dtype.names or ()
    required = {"x", "y", "z", "scalar_Intensity", "scalar_Label"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{scan.name} is missing required vertex fields: {sorted(missing)}")

    records = np.memmap(scan, mode="r", dtype=dtype, offset=payload_offset, shape=(vertex_count,))
    x_min = float(np.min(records["x"]))
    y_min = float(np.min(records["y"]))
    x_max = float(np.max(records["x"]))
    y_max = float(np.max(records["y"]))
    n_cols = max(1, math.ceil((x_max - x_min) / tile_size))
    n_rows = max(1, math.ceil((y_max - y_min) / tile_size))
    n_tiles = n_cols * n_rows

    # A uint32 tile ID per source point is disk-backed. Sorting its indices groups
    # points by tile in one pass, instead of rescanning a 20M+ point cloud per tile.
    tile_ids_path = x_path.with_suffix(".tile_ids.tmp")
    order_path = x_path.with_suffix(".order.tmp")
    tile_ids = np.memmap(tile_ids_path, mode="w+", dtype=np.uint32, shape=(vertex_count,))
    chunk_size = 1_000_000
    for start in range(0, vertex_count, chunk_size):
        stop = min(start + chunk_size, vertex_count)
        col = np.floor((records["x"][start:stop] - x_min) / tile_size).astype(np.int64)
        row = np.floor((records["y"][start:stop] - y_min) / tile_size).astype(np.int64)
        col = np.clip(col, 0, n_cols - 1)
        row = np.clip(row, 0, n_rows - 1)
        tile_ids[start:stop] = (row * n_cols + col).astype(np.uint32)
    tile_ids.flush()

    order = np.argsort(tile_ids, kind="stable").astype(np.uint32, copy=False)
    # Keep the sorted index on disk too, so tile extraction doesn't need another
    # full-size resident copy after sorting.
    order_map = np.memmap(order_path, mode="w+", dtype=np.uint32, shape=(vertex_count,))
    order_map[:] = order
    order_map.flush()
    del order, order_map, tile_ids
    tile_ids = np.memmap(tile_ids_path, mode="r", dtype=np.uint32, shape=(vertex_count,))
    order_map = np.memmap(order_path, mode="r", dtype=np.uint32, shape=(vertex_count,))

    counts = np.bincount(np.asarray(tile_ids), minlength=n_tiles)
    offsets = np.empty(n_tiles + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    valid_tiles = np.flatnonzero(counts >= min_points)
    x_out = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                      shape=(len(valid_tiles), 4, n_points))
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64,
                                      shape=(len(valid_tiles), n_points))
    rng = np.random.default_rng(seed)

    for out_i, tile_id in enumerate(valid_tiles):
        lo, hi = offsets[tile_id], offsets[tile_id + 1]
        member_indices = order_map[lo:hi]
        if len(member_indices) > n_points:
            chosen = rng.choice(len(member_indices), size=n_points, replace=False)
            selected = member_indices[chosen]
        else:
            selected = rng.choice(member_indices, size=n_points, replace=True)

        px = np.asarray(records["x"][selected], dtype=np.float32)
        py = np.asarray(records["y"][selected], dtype=np.float32)
        pz = np.asarray(records["z"][selected], dtype=np.float32)
        intensity = np.asarray(records["scalar_Intensity"][selected], dtype=np.float32)
        # Some Toronto_3D returns have NaN intensity despite valid coordinates and labels.
        # Map missing/non-finite returns to zero before scaling so model inputs remain finite.
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=255.0, neginf=0.0)
        label_values = np.rint(records["scalar_Label"][selected]).astype(np.int64)
        if np.any((label_values < 0) | (label_values >= NUM_TORONTO_CLASSES)):
            raise ValueError(f"Unexpected semantic label ID in {scan.name}, tile {tile_id}")

        row, col = divmod(int(tile_id), n_cols)
        tile_center_x = min(x_min + (col + 0.5) * tile_size, x_max)
        tile_center_y = min(y_min + (row + 0.5) * tile_size, y_max)
        # Keep preprocessing available at inference, where semantic labels are unknown.
        z_reference = float(np.median(records["z"][member_indices]))

        x_out[out_i, 0] = (px - tile_center_x) / coord_scale
        x_out[out_i, 1] = (py - tile_center_y) / coord_scale
        x_out[out_i, 2] = (pz - z_reference) / coord_scale
        # Toronto_3D stores intensity on a 1..255 scale; model input uses ~0..1.
        x_out[out_i, 3] = np.clip(intensity / 255.0, 0.0, 1.0)
        y_out[out_i] = label_values

    x_out.flush()
    y_out.flush()
    metadata = {
        "scan": str(scan.resolve()),
        "source_points": vertex_count,
        "tile_count": len(valid_tiles),
        "tile_size_m": tile_size,
        "n_points": n_points,
        "min_points": min_points,
        "coord_scale": coord_scale,
        "num_classes": NUM_TORONTO_CLASSES,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    del records, tile_ids, order_map, x_out, y_out
    tile_ids_path.unlink(missing_ok=True)
    order_path.unlink(missing_ok=True)


def load_toronto_scan_for_inference(
    scan: str | Path,
    n_points: int = 2048,
    tile_size: float = 20.0,
    coord_scale: float = 25.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create label-free model tiles and source-point indices for one Toronto PLY scan.

    Returns ``(features, point_indices, labels)``. ``labels`` is ``None`` when the
    source PLY has no ``scalar_Label`` property. Each tile has exactly ``n_points``;
    dense tiles are sampled without replacement and sparse tiles with replacement.
    """
    scan = Path(scan)
    if n_points <= 0 or tile_size <= 0 or coord_scale <= 0:
        raise ValueError("n_points, tile_size and coord_scale must be positive")
    header, payload_offset = _read_header(scan)
    vertex_count, dtype = _vertex_layout(header)
    names = dtype.names or ()
    required = {"x", "y", "z", "scalar_Intensity"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{scan.name} is missing required vertex fields: {sorted(missing)}")

    records = np.memmap(scan, mode="r", dtype=dtype, offset=payload_offset, shape=(vertex_count,))
    px_all = records["x"]
    py_all = records["y"]
    x_min, x_max = float(np.min(px_all)), float(np.max(px_all))
    y_min, y_max = float(np.min(py_all)), float(np.max(py_all))
    n_cols = max(1, math.ceil((x_max - x_min) / tile_size))
    n_rows = max(1, math.ceil((y_max - y_min) / tile_size))
    cols = np.clip(np.floor((px_all - x_min) / tile_size).astype(np.int64), 0, n_cols - 1)
    rows = np.clip(np.floor((py_all - y_min) / tile_size).astype(np.int64), 0, n_rows - 1)
    tile_ids = rows * n_cols + cols
    order = np.argsort(tile_ids, kind="stable")
    counts = np.bincount(tile_ids, minlength=n_cols * n_rows)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    features, indices, labels = [], [], []
    has_labels = "scalar_Label" in names
    rng = np.random.default_rng(seed)
    for tile_id in np.flatnonzero(counts):
        members = order[offsets[tile_id]:offsets[tile_id + 1]]
        row, col = divmod(int(tile_id), n_cols)
        x_center = min(x_min + (col + 0.5) * tile_size, x_max)
        y_center = min(y_min + (row + 0.5) * tile_size, y_max)
        z_reference = float(np.median(records["z"][members]))
        for start in range(0, len(members), n_points):
            selected = members[start:start + n_points]
            if len(selected) < n_points:
                selected = np.concatenate((selected, rng.choice(selected, n_points - len(selected), replace=True)))
            px = np.asarray(records["x"][selected], dtype=np.float32)
            py = np.asarray(records["y"][selected], dtype=np.float32)
            pz = np.asarray(records["z"][selected], dtype=np.float32)
            intensity = np.nan_to_num(
                np.asarray(records["scalar_Intensity"][selected], dtype=np.float32),
                nan=0.0, posinf=255.0, neginf=0.0,
            )
            tile_features = np.stack((
                (px - x_center) / coord_scale,
                (py - y_center) / coord_scale,
                (pz - z_reference) / coord_scale,
                np.clip(intensity / 255.0, 0.0, 1.0),
            )).astype(np.float32)
            if not np.isfinite(tile_features).all():
                raise ValueError(f"Non-finite features generated for {scan.name}")
            features.append(tile_features)
            indices.append(selected.astype(np.int64))
            if has_labels:
                labels.append(np.rint(records["scalar_Label"][selected]).astype(np.int64))
    if not features:
        raise ValueError(f"No points/tiles found in {scan}")
    del records
    return (np.stack(features), np.stack(indices),
            np.stack(labels) if has_labels else None)


def merge_tile_predictions(point_indices: np.ndarray, tile_logits: np.ndarray,
                           n_source_points: int, n_classes: int) -> np.ndarray:
    """Average sampled tile logits back onto source points and return per-point labels."""
    scores = np.zeros((n_source_points, n_classes), dtype=np.float64)
    counts = np.zeros(n_source_points, dtype=np.int64)
    for indices, logits in zip(point_indices, tile_logits):
        np.add.at(scores, indices, logits)
        np.add.at(counts, indices, 1)
    covered = counts > 0
    scores[covered] /= counts[covered, None]
    return scores.argmax(axis=1).astype(np.int64)


class Toronto3DTiles(Dataset):
    """Fixed-size XYZ+intensity tiles with per-point Toronto_3D semantic labels.

    Parameters
    ----------
    scans:
        A PLY path, directory containing ``L*.ply``, or sequence of PLY paths.
    n_points:
        Number of points per tile returned, matching the model's static point dimension.
    tile_size:
        XY tile width in metres. Overlapping tiles are not generated.
    min_points:
        Drop tiles with fewer source points than this threshold.
    coord_scale:
        Normalize centered XYZ coordinates by this number of metres.
    augment:
        Apply random yaw rotation, mild XYZ scaling/jitter, and point permutation on access.
    cache_dir:
        Preprocessed fixed-size samples are cached here and reused on later runs.
    """

    def __init__(self, scans: str | Path | list[str | Path] | tuple[str | Path, ...],
                 n_points: int = 2048, tile_size: float = 20.0, min_points: int = 2048,
                 coord_scale: float = 25.0, augment: bool = False,
                 cache_dir: str | Path = "outputs/toronto_cache", seed: int = 0):
        if n_points <= 0 or tile_size <= 0 or min_points <= 0 or coord_scale <= 0:
            raise ValueError("n_points, tile_size, min_points and coord_scale must be positive")
        if isinstance(scans, (str, Path)):
            source = Path(scans)
            paths = sorted(source.glob("L*.ply")) if source.is_dir() else [source]
        else:
            paths = [Path(path) for path in scans]
        paths = [path.resolve() for path in paths]
        if not paths:
            raise FileNotFoundError(f"No Toronto_3D PLY scans found from: {scans}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        self.n_points = n_points
        self.augment = augment
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[tuple[Path, Path, int]] = []
        self._x_maps: dict[Path, np.ndarray] = {}
        self._y_maps: dict[Path, np.ndarray] = {}
        self.class_counts = np.zeros(NUM_TORONTO_CLASSES, dtype=np.int64)

        for scan_i, scan in enumerate(paths):
            x_path, y_path, meta_path = _cache_paths(scan, self.cache_dir, tile_size,
                                                      n_points, min_points, coord_scale)
            if not (x_path.exists() and y_path.exists() and meta_path.exists()):
                _build_scan_cache(scan, x_path, y_path, meta_path, tile_size,
                                  n_points, min_points, coord_scale, seed + scan_i)
            metadata = json.loads(meta_path.read_text())
            n_tiles = int(metadata["tile_count"])
            if n_tiles == 0:
                continue
            self._entries.append((x_path, y_path, n_tiles))
            labels = np.load(y_path, mmap_mode="r")
            self.class_counts += np.bincount(labels.reshape(-1), minlength=NUM_TORONTO_CLASSES)
            del labels
        if not self._entries:
            raise ValueError("No tiles met min_points; try a larger tile_size or lower min_points")
        self._ends = np.cumsum([entry[2] for entry in self._entries])
        self.total_tiles = int(self._ends[-1])

    def __len__(self) -> int:
        return self.total_tiles

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total_tiles
        if index < 0 or index >= self.total_tiles:
            raise IndexError(index)
        file_i = int(np.searchsorted(self._ends, index, side="right"))
        start = 0 if file_i == 0 else int(self._ends[file_i - 1])
        local_i = index - start
        x_path, y_path, _ = self._entries[file_i]
        if x_path not in self._x_maps:
            self._x_maps[x_path] = np.load(x_path, mmap_mode="r")
            self._y_maps[y_path] = np.load(y_path, mmap_mode="r")
        x = np.array(self._x_maps[x_path][local_i], dtype=np.float32, copy=True)
        y = np.array(self._y_maps[y_path][local_i], dtype=np.int64, copy=True)

        if self.augment:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            x[[0, 1]] = np.stack((c * x[0] - s * x[1], s * x[0] + c * x[1]))
            scale = np.random.uniform(0.95, 1.05)
            x[:3] *= scale
            x[:3] += np.random.normal(0.0, 0.002, size=x[:3].shape).astype(np.float32)
            permutation = np.random.permutation(self.n_points)
            x = x[:, permutation]
            y = y[permutation]
        return torch.from_numpy(x), torch.from_numpy(y)
