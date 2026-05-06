from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphSample:
    sample_id: str
    x_face: torch.Tensor
    gt_ff_adj: torch.Tensor
    gt_space_exists: torch.Tensor
    gt_sf_adj: torch.Tensor


def parse_building_labels(path: Path, max_spaces: int | None = None):
    with np.load(path, allow_pickle=True) as payload:
        face_feats = np.asarray(payload["face_feats"], dtype=np.float32)
        space_feats = np.asarray(payload["space_feats"], dtype=np.float32)
        ff_edges = np.asarray(payload["ff_edges"], dtype=np.int64)
        sf_edges = np.asarray(payload["sf_edges"], dtype=np.int64)

    n_faces = int(face_feats.shape[0])
    n_spaces_total = int(space_feats.shape[0])

    gt_ff_adj = np.zeros((n_faces, n_faces), dtype=np.float32)
    for i, j in ff_edges:
        if 0 <= int(i) < n_faces and 0 <= int(j) < n_faces and int(i) != int(j):
            ii = int(i)
            jj = int(j)
            gt_ff_adj[ii, jj] = 1.0
            gt_ff_adj[jj, ii] = 1.0

    # PACK building sf_edges are stored as (face_idx, space_idx).
    gt_sf_full = np.zeros((n_spaces_total, n_faces), dtype=np.float32)
    for face_idx, space_idx in sf_edges:
        fi = int(face_idx)
        si = int(space_idx)
        if 0 <= fi < n_faces and 0 <= si < n_spaces_total:
            gt_sf_full[si, fi] = 1.0

    if max_spaces is None:
        n_slots = n_spaces_total
    else:
        n_slots = int(max_spaces)

    n_used = min(n_slots, n_spaces_total)
    gt_space_exists = np.zeros((n_slots,), dtype=np.float32)
    if n_used > 0:
        gt_space_exists[:n_used] = 1.0

    gt_sf_adj = np.zeros((n_slots, n_faces), dtype=np.float32)
    if n_used > 0:
        gt_sf_adj[:n_used] = gt_sf_full[:n_used]

    return n_faces, gt_ff_adj, gt_space_exists, gt_sf_adj


def ids_from_building(building_root: Path) -> List[str]:
    return sorted([p.stem for p in building_root.glob("*.npz")])


def ids_from_graph(graph_root: Path) -> List[str]:
    # Backward-compatible alias for older imports.
    return ids_from_building(graph_root)


def _normalize_split(raw: str) -> str | None:
    text = str(raw).strip().lower()
    if text in {"train", "tr", "training"}:
        return "train"
    if text in {"val", "valid", "validation", "dev"}:
        return "val"
    if text in {"test", "te", "testing"}:
        return "test"
    return None


def _dedupe_keep_order(ids: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        result.append(sid)
    return result


def build_sample_splits_from_csv(
    split_csv_path: str | Path,
    allowed_ids: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    csv_path = Path(split_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV is empty: {csv_path}")

        colmap = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
        split_col = colmap.get("split")
        id_col = colmap.get("building_id")

        if split_col is None:
            raise ValueError(f"Split CSV missing required column: split ({csv_path})")
        if id_col is None:
            raise ValueError(
                f"Split CSV missing required column: building_id ({csv_path})"
            )

        raw_splits = {"train": [], "val": [], "test": []}
        for row in reader:
            sid = str(row.get(id_col, "")).strip()
            split_name = _normalize_split(row.get(split_col, ""))
            if not sid or split_name is None:
                continue
            raw_splits[split_name].append(sid)

    train_ids = _dedupe_keep_order(raw_splits["train"])
    val_ids = _dedupe_keep_order(raw_splits["val"])
    test_ids = _dedupe_keep_order(raw_splits["test"])

    train_set = set(train_ids)
    val_ids = [sid for sid in val_ids if sid not in train_set]
    val_set = set(val_ids)
    test_ids = [sid for sid in test_ids if sid not in train_set and sid not in val_set]

    if allowed_ids is not None:
        allow = set(str(sid) for sid in allowed_ids)
        train_ids = [sid for sid in train_ids if sid in allow]
        val_ids = [sid for sid in val_ids if sid in allow]
        test_ids = [sid for sid in test_ids if sid in allow]

    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            "Split CSV resolved to empty train/val/test after filtering. "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    return {"train": train_ids, "val": val_ids, "test": test_ids}


class GraphToTopologyDataset(Dataset):
    def __init__(
        self,
        building_root: str | Path,
        geometry_root: str | Path,
        sample_ids: Sequence[str],
        max_spaces: int | None = None,
        face_feat_dim: int | None = None,
    ):
        self.building_root = Path(building_root)
        self.geometry_root = Path(geometry_root)
        self.sample_ids = list(sample_ids)
        self.max_spaces = max_spaces

        if len(self.sample_ids) > 0:
            sid = self.sample_ids[0]
            geom_path = self.geometry_root / f"{sid}.npz"
            if not geom_path.exists():
                raise FileNotFoundError(f"Geometry npz not found: {geom_path}")
            with np.load(geom_path, allow_pickle=True) as geom:
                if "x_face_scr" not in geom:
                    raise KeyError(f"Missing 'x_face_scr' in {geom_path}. Please regenerate geometry npz.")
                inferred_dim = int(geom["x_face_scr"].shape[1])
            self.face_feat_dim = int(face_feat_dim) if face_feat_dim is not None else inferred_dim
        else:
            self.face_feat_dim = int(face_feat_dim) if face_feat_dim is not None else 32

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> GraphSample:
        sid = self.sample_ids[idx]
        geom_path = self.geometry_root / f"{sid}.npz"
        building_path = self.building_root / f"{sid}.npz"

        if not geom_path.exists():
            raise FileNotFoundError(f"Geometry npz not found: {geom_path}")
        if not building_path.exists():
            raise FileNotFoundError(f"Building npz not found: {building_path}")

        with np.load(geom_path, allow_pickle=True) as geom:
            if "x_face_scr" not in geom:
                raise KeyError(f"Missing 'x_face_scr' in {geom_path}. Please regenerate geometry npz.")
            x_face = np.asarray(geom["x_face_scr"], dtype=np.float32)
            if x_face.shape[1] != self.face_feat_dim:
                raise ValueError(
                    f"Feature dim mismatch for {sid}: geometry has {x_face.shape[1]}, expected {self.face_feat_dim}."
                )
            face_names_geom = [str(v) for v in geom.get("face_names", np.arange(x_face.shape[0])).tolist()]

        n_faces_building, gt_ff_adj, gt_space_exists, gt_sf_adj = parse_building_labels(
            path=building_path,
            max_spaces=self.max_spaces,
        )

        if n_faces_building != int(x_face.shape[0]):
            raise ValueError(
                f"Face count mismatch for {sid}: building has {n_faces_building}, geometry has {x_face.shape[0]}."
            )
        if len(face_names_geom) == n_faces_building:
            expected = [str(i) for i in range(n_faces_building)]
            if any(str(a) != str(b) for a, b in zip(face_names_geom, expected)):
                raise ValueError(f"Face order mismatch between building and geometry for sample {sid}.")

        return GraphSample(
            sample_id=sid,
            x_face=torch.from_numpy(x_face),
            gt_ff_adj=torch.from_numpy(gt_ff_adj),
            gt_space_exists=torch.from_numpy(gt_space_exists),
            gt_sf_adj=torch.from_numpy(gt_sf_adj),
        )
