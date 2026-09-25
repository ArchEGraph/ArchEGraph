#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the sensitivity of geometric face-face topology to the "
            "shared-boundary threshold tau_e."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    parser.add_argument(
        "--split-files",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Mesh split CSVs used as cohorts. Defaults to split_p_mesh.csv and "
            "split_m_mesh.csv under DATA_DIR/split."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--reference-threshold", type=float, default=0.1)
    parser.add_argument(
        "--direction-decimals",
        type=int,
        default=5,
        help="Decimal precision used to group parallel boundary segments.",
    )
    parser.add_argument(
        "--offset-decimals",
        type=int,
        default=3,
        help="Decimal precision used to group collinear boundary segments.",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=None,
        help="Optional per-cohort limit for a quick validation run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to cache/tau_e_sensitivity_TIMESTAMP.",
    )
    return parser.parse_args()


def _threshold_key(value: float) -> str:
    return f"{value:g}"


def _canonical_line(
    start: np.ndarray,
    end: np.ndarray,
    direction_decimals: int,
    offset_decimals: int,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return None

    direction = direction / length
    nonzero = np.flatnonzero(np.abs(direction) > 1e-8)
    if nonzero.size == 0:
        return None
    if direction[int(nonzero[0])] < 0:
        direction = -direction

    offset = start - float(np.dot(start, direction)) * direction
    direction_key = tuple(float(value) for value in np.round(direction, direction_decimals))
    offset_key = tuple(float(value) for value in np.round(offset, offset_decimals))
    return direction_key, offset_key


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1e-8:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _intersection_length(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    first_index = 0
    second_index = 0
    total = 0.0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        total += max(0.0, min(first_end, second_end) - max(first_start, second_start))
        if first_end < second_end:
            first_index += 1
        else:
            second_index += 1
    return total


def _shared_boundary_lengths(
    face_vertices: np.ndarray,
    direction_decimals: int,
    offset_decimals: int,
) -> dict[tuple[int, int], float]:
    line_segments: dict[
        tuple[tuple[float, ...], tuple[float, ...]],
        dict[int, list[tuple[float, float]]],
    ] = defaultdict(lambda: defaultdict(list))

    for face_index, flattened_vertices in enumerate(face_vertices):
        vertices = np.asarray(flattened_vertices, dtype=np.float64).reshape(-1, 3)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            line_key = _canonical_line(
                start,
                end,
                direction_decimals=direction_decimals,
                offset_decimals=offset_decimals,
            )
            if line_key is None:
                continue
            direction = np.asarray(line_key[0], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            interval = sorted((float(np.dot(start, direction)), float(np.dot(end, direction))))
            line_segments[line_key][face_index].append((interval[0], interval[1]))

    shared_lengths: dict[tuple[int, int], float] = defaultdict(float)
    for segments_by_face in line_segments.values():
        if len(segments_by_face) < 2:
            continue
        merged_by_face = {
            face_index: _merge_intervals(intervals)
            for face_index, intervals in segments_by_face.items()
        }
        face_indices = sorted(merged_by_face)
        for first_position, first_face in enumerate(face_indices):
            for second_face in face_indices[first_position + 1 :]:
                overlap = _intersection_length(
                    merged_by_face[first_face],
                    merged_by_face[second_face],
                )
                if overlap > 1e-8:
                    shared_lengths[(first_face, second_face)] += overlap
    return dict(shared_lengths)


def _undirected_edges(edges: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(source), int(destination))))
        for source, destination in edges
        if int(source) != int(destination)
    }


def _load_cohorts(data_dir: Path, split_files: list[Path] | None) -> dict[str, list[str]]:
    if split_files is None:
        split_files = [
            data_dir / "split" / "split_p_mesh.csv",
            data_dir / "split" / "split_m_mesh.csv",
        ]

    cohorts: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for split_file in split_files:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        cohort_name = split_file.stem.removeprefix("split_").removesuffix("_mesh").upper()
        building_ids = []
        with split_file.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                building_id = str(row.get("building_id", "")).strip()
                if building_id and building_id not in seen_ids:
                    building_ids.append(building_id)
                    seen_ids.add(building_id)
        if not building_ids:
            raise ValueError(f"No building_id values found in {split_file}")
        cohorts[cohort_name] = building_ids
    return cohorts


def _new_aggregate() -> dict[str, float]:
    return {
        "buildings": 0,
        "edges": 0,
        "reference_edges": 0,
        "reference_intersection": 0,
        "reference_union": 0,
        "unchanged_buildings": 0,
        "released_edges": 0,
        "released_intersection": 0,
    }


def _update_aggregate(
    aggregate: dict[str, float],
    edges: set[tuple[int, int]],
    reference_edges: set[tuple[int, int]],
    released_edges: set[tuple[int, int]],
) -> None:
    aggregate["buildings"] += 1
    aggregate["edges"] += len(edges)
    aggregate["reference_edges"] += len(reference_edges)
    aggregate["reference_intersection"] += len(edges & reference_edges)
    aggregate["reference_union"] += len(edges | reference_edges)
    aggregate["unchanged_buildings"] += int(edges == reference_edges)
    aggregate["released_edges"] += len(released_edges)
    aggregate["released_intersection"] += len(edges & released_edges)


def _finalize_aggregate(aggregate: dict[str, float]) -> dict[str, float | int]:
    buildings = int(aggregate["buildings"])
    edges = int(aggregate["edges"])
    reference_edges = int(aggregate["reference_edges"])
    reference_intersection = int(aggregate["reference_intersection"])
    reference_union = int(aggregate["reference_union"])
    released_edges = int(aggregate["released_edges"])
    released_intersection = int(aggregate["released_intersection"])

    return {
        "buildings": buildings,
        "edges": edges,
        "mean_edges_per_building": edges / max(buildings, 1),
        "edge_delta_vs_reference_pct": 100.0 * (edges - reference_edges) / max(reference_edges, 1),
        "edge_f1_vs_reference": 2.0 * reference_intersection / max(edges + reference_edges, 1),
        "edge_jaccard_vs_reference": reference_intersection / max(reference_union, 1),
        "unchanged_buildings_pct": 100.0 * aggregate["unchanged_buildings"] / max(buildings, 1),
        "precision_vs_released": released_intersection / max(edges, 1),
        "recall_vs_released": released_intersection / max(released_edges, 1),
        "edge_f1_vs_released": 2.0 * released_intersection / max(edges + released_edges, 1),
    }


def _analyze(
    data_dir: Path,
    cohorts: dict[str, list[str]],
    thresholds: list[float],
    reference_threshold: float,
    direction_decimals: int,
    offset_decimals: int,
    max_buildings: int | None,
) -> dict[str, object]:
    threshold_keys = [_threshold_key(value) for value in thresholds]
    aggregates = {
        cohort: {key: _new_aggregate() for key in threshold_keys}
        for cohort in [*cohorts, "ALL"]
    }

    for cohort, all_building_ids in cohorts.items():
        building_ids = all_building_ids[:max_buildings] if max_buildings else all_building_ids
        for position, building_id in enumerate(building_ids, start=1):
            geometry_path = data_dir / "geometry" / f"{building_id}.npz"
            building_path = data_dir / "building" / f"{building_id}.npz"
            with np.load(geometry_path, allow_pickle=True) as geometry:
                shared_lengths = _shared_boundary_lengths(
                    geometry["face_v"],
                    direction_decimals=direction_decimals,
                    offset_decimals=offset_decimals,
                )
            with np.load(building_path) as building:
                released_edges = _undirected_edges(building["ff_edges"])

            edges_by_threshold = {
                threshold: {
                    pair for pair, shared_length in shared_lengths.items() if shared_length > threshold
                }
                for threshold in thresholds
            }
            reference_edges = edges_by_threshold[reference_threshold]
            for threshold, edges in edges_by_threshold.items():
                key = _threshold_key(threshold)
                _update_aggregate(aggregates[cohort][key], edges, reference_edges, released_edges)
                _update_aggregate(aggregates["ALL"][key], edges, reference_edges, released_edges)

            if position % 100 == 0 or position == len(building_ids):
                print(f"[{cohort}] processed {position}/{len(building_ids)} buildings", flush=True)

    summary = {
        cohort: {
            threshold: _finalize_aggregate(values)
            for threshold, values in threshold_values.items()
        }
        for cohort, threshold_values in aggregates.items()
    }
    return {
        "protocol": {
            "thresholds": thresholds,
            "reference_threshold": reference_threshold,
            "edge_rule": "cumulative collinear boundary overlap length > tau_e",
            "direction_decimals": direction_decimals,
            "offset_decimals": offset_decimals,
            "cohorts": {name: len(ids[:max_buildings] if max_buildings else ids) for name, ids in cohorts.items()},
            "note": (
                "Released ff_edges include downstream enclosure reconstruction and rule-based "
                "topological refinement; agreement with released labels is therefore diagnostic, "
                "not expected to be exact."
            ),
        },
        "summary": summary,
    }


def _write_outputs(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    rows = []
    for cohort, thresholds in payload["summary"].items():
        for threshold, metrics in thresholds.items():
            rows.append({"cohort": cohort, "tau_e": threshold, **metrics})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_summary = payload["summary"]["ALL"]
    markdown = [
        "| $\\tau_e$ | Edges | $\\Delta E$ vs. ref. | Edge F1 vs. ref. | Jaccard vs. ref. | Unchanged buildings |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, metrics in all_summary.items():
        markdown.append(
            f"| {threshold} | {metrics['edges']:,} | "
            f"{metrics['edge_delta_vs_reference_pct']:+.2f}\\% | "
            f"{metrics['edge_f1_vs_reference']:.4f} | "
            f"{metrics['edge_jaccard_vs_reference']:.4f} | "
            f"{metrics['unchanged_buildings_pct']:.2f}\\% |"
        )
    with (output_dir / "rebuttal_table.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    thresholds = sorted(set(float(value) for value in args.thresholds))
    reference_threshold = float(args.reference_threshold)
    if any(value < 0 for value in thresholds):
        raise ValueError("All thresholds must be non-negative.")
    if reference_threshold not in thresholds:
        raise ValueError("reference-threshold must also appear in thresholds.")

    split_files = (
        [path.expanduser().resolve() for path in args.split_files]
        if args.split_files is not None
        else None
    )
    cohorts = _load_cohorts(data_dir, split_files)
    payload = _analyze(
        data_dir=data_dir,
        cohorts=cohorts,
        thresholds=thresholds,
        reference_threshold=reference_threshold,
        direction_decimals=args.direction_decimals,
        offset_decimals=args.offset_decimals,
        max_buildings=args.max_buildings,
    )

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT_DIR / "cache" / f"tau_e_sensitivity_{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    _write_outputs(output_dir, payload)
    print(f"Wrote tau_e sensitivity results to {output_dir}")


if __name__ == "__main__":
    main()