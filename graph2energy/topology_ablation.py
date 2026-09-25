from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any


TOPOLOGY_MODES = ("original", "shuffled")


def _building_key(case_key: str) -> str:
    return str(case_key).split("__", maxsplit=1)[0]


def _building_seed(seed: int, building_key: str) -> int:
    payload = f"{int(seed)}:{building_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _shuffle_targets(targets: list[int], seed: int) -> list[int]:
    shuffled = list(targets)
    random.Random(seed).shuffle(shuffled)

    if shuffled == targets and len(set(targets)) > 1:
        for offset in range(1, len(targets)):
            candidate = targets[offset:] + targets[:offset]
            if candidate != targets:
                return candidate
    return shuffled


def _randomize_edges(
    edges: list[list[int]],
    n_spaces: int,
    seed: int,
) -> tuple[list[list[int]], int]:
    if any(len(edge) != 2 for edge in edges):
        raise ValueError("Each face-to-space edge must contain exactly two indices.")
    normalized = [[int(edge[0]), int(edge[1])] for edge in edges]

    targets = [edge[1] for edge in normalized]
    invalid_targets = [target for target in targets if target < 0 or target >= n_spaces]
    if invalid_targets:
        raise ValueError(
            f"Face-to-space targets outside [0, {n_spaces}): {sorted(set(invalid_targets))}"
        )

    shuffled_targets = _shuffle_targets(targets, seed=seed)
    if Counter(shuffled_targets) != Counter(targets):
        raise RuntimeError("Topology shuffle changed the per-space degree distribution.")

    randomized = [
        [source, shuffled_targets[index]]
        for index, (source, _) in enumerate(normalized)
    ]
    changed_edges = sum(before != after for before, after in zip(normalized, randomized))
    return randomized, changed_edges


def apply_topology_ablation(
    data_dict: dict[str, dict[str, Any]],
    mode: str = "original",
    seed: int = 42,
) -> dict[str, int | str]:
    """Apply one fixed face-to-space topology condition to loaded G2E cases.

    ``shuffled`` permutes only destination space indices within each building.
    It preserves source faces, edge order, edge attributes, total edge count,
    and the number of incident faces per space. Cases of the same building use
    the same randomized topology regardless of weather location.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in TOPOLOGY_MODES:
        raise ValueError(
            f"Unsupported topology mode: {mode}. Expected one of: {', '.join(TOPOLOGY_MODES)}"
        )

    stats: dict[str, int | str] = {
        "mode": normalized_mode,
        "seed": int(seed),
        "cases": len(data_dict),
        "buildings": 0,
        "edges": 0,
        "changed_edges": 0,
        "single_space_buildings": 0,
    }

    randomized_by_building: dict[
        str, tuple[tuple[tuple[int, int], ...], list[list[int]], int]
    ] = {}

    for case_key, case in data_dict.items():
        if not isinstance(case, dict) or not isinstance(case.get("building"), dict):
            raise TypeError(
                "Topology ablation expects raw G2E cases with a 'building' dictionary."
            )

        building = case["building"]
        edges = building.get("sf_edges")
        space_features = building.get("space_feats")
        if edges is None or space_features is None:
            raise ValueError(
                f"Case {case_key} is missing building sf_edges or space_feats."
            )

        if any(len(edge) != 2 for edge in edges):
            raise ValueError(
                f"Case {case_key} contains a face-to-space edge without two indices."
            )
        normalized_edges = [[int(edge[0]), int(edge[1])] for edge in edges]
        signature = tuple(tuple(edge) for edge in normalized_edges)
        building_key = _building_key(case_key)

        cached = randomized_by_building.get(building_key)
        if cached is None:
            n_spaces = len(space_features)
            if normalized_mode == "shuffled":
                randomized, changed_edges = _randomize_edges(
                    normalized_edges,
                    n_spaces=n_spaces,
                    seed=_building_seed(seed, building_key),
                )
            else:
                randomized = [list(edge) for edge in normalized_edges]
                changed_edges = 0
            randomized_by_building[building_key] = (
                signature,
                randomized,
                changed_edges,
            )
            stats["buildings"] += 1
            stats["edges"] += len(normalized_edges)
            stats["changed_edges"] += changed_edges
            if n_spaces <= 1:
                stats["single_space_buildings"] += 1
        else:
            cached_signature, randomized, _ = cached
            if signature != cached_signature:
                raise ValueError(
                    f"Building {building_key} has inconsistent topology across weather cases."
                )

        if normalized_mode == "shuffled":
            building["sf_edges"] = [list(edge) for edge in randomized]

    return stats