#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path


SPLIT_FIELDS = [
    "case_id",
    "sample_id",
    "building_id",
    "weather_id",
    "subset",
    "split",
    "scenario",
]

WEATHER_PARTITIONS = {
    "train": [
        "Accra",
        "Beijing",
        "Capetown",
        "Chicago",
        "Delhi",
        "Kualalumpur",
        "Lagos",
        "Manila",
        "Mexicocity",
        "Mumbai",
        "Nairobi",
        "Prague",
        "Vienna",
        "Warsaw",
    ],
    "val": ["Quito", "Saopaulo", "Seoul"],
    "test": ["Anchorage", "Tokyo", "Vancouver"],
}


def _normalize_building_id(value: str) -> str:
    return str(int(str(value).strip()))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _serialize_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SPLIT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _subset_map(split_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename, subset in (("split_p.csv", "P"), ("split_m.csv", "M")):
        for row in _read_csv(split_dir / filename):
            building_id = _normalize_building_id(row["building_id"])
            previous = result.setdefault(building_id, subset)
            if previous != subset:
                raise ValueError(f"Building {building_id} appears in both P and M")
    return result


def _manifest_index(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    by_sample: dict[str, dict[str, str]] = {}
    weather_by_building: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv(manifest_path):
        sample_id = row["sample_id"].strip()
        building_id = _normalize_building_id(row["building_id"])
        weather_id = row["weather_id"].strip()
        by_sample[sample_id] = row
        weather_by_building[building_id].add(weather_id)
    return by_sample, weather_by_building


def _selected_buildings(rows: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for row in rows:
        building_id = _normalize_building_id(row["building_id"])
        if building_id not in seen:
            seen.add(building_id)
            selected.append(building_id)
    return selected


def _repair_building_split(
    rows: list[dict[str, str]],
    subset_by_building: dict[str, str],
    manifest_by_sample: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    repaired = []
    for row in rows:
        sample_id = row["sample_id"].strip()
        building_id = _normalize_building_id(row["building_id"])
        if sample_id not in manifest_by_sample:
            raise ValueError(f"Building split sample is absent from manifest: {sample_id}")
        repaired.append(
            {
                "case_id": sample_id,
                "sample_id": sample_id,
                "building_id": building_id,
                "weather_id": row["weather_id"].strip(),
                "subset": subset_by_building[building_id],
                "split": row["split"].strip().lower(),
                "scenario": "building-bias",
            }
        )
    return repaired


def _generate_weather_split(
    selected_buildings: list[str],
    subset_by_building: dict[str, str],
    manifest_by_sample: dict[str, dict[str, str]],
    weather_by_building: dict[str, set[str]],
) -> list[dict[str, str]]:
    required_weather = {
        weather for weather_ids in WEATHER_PARTITIONS.values() for weather in weather_ids
    }
    for building_id in selected_buildings:
        missing = required_weather - weather_by_building[building_id]
        if missing:
            raise ValueError(
                f"Building {building_id} lacks required weather cases: {sorted(missing)}"
            )

    rows = []
    for split_name, weather_ids in WEATHER_PARTITIONS.items():
        for building_id in selected_buildings:
            padded_id = f"{int(building_id):05d}"
            for weather_id in weather_ids:
                sample_id = f"{padded_id}__{weather_id}"
                if sample_id not in manifest_by_sample:
                    raise ValueError(f"Weather split sample is absent from manifest: {sample_id}")
                rows.append(
                    {
                        "case_id": sample_id,
                        "sample_id": sample_id,
                        "building_id": building_id,
                        "weather_id": weather_id,
                        "subset": subset_by_building[building_id],
                        "split": split_name,
                        "scenario": "weather-bias",
                    }
                )
    return rows


def _values_by_split(rows: list[dict[str, str]], field: str) -> dict[str, set[str]]:
    result = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        result[row["split"]].add(row[field])
    return result


def _assert_disjoint(values: dict[str, set[str]], label: str) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = values[left] & values[right]
        if overlap:
            raise ValueError(f"{label} overlap between {left}/{right}: {sorted(overlap)}")


def _validate(
    building_rows: list[dict[str, str]], weather_rows: list[dict[str, str]]
) -> None:
    if len(building_rows) != 3000 or len(weather_rows) != 3000:
        raise ValueError(
            f"Expected 3,000 rows per OOD split, got {len(building_rows)} and {len(weather_rows)}"
        )

    for label, rows in (("building-OOD", building_rows), ("weather-OOD", weather_rows)):
        case_ids = [row["case_id"] for row in rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Duplicate case IDs in {label} split")

    building_counts = defaultdict(int)
    weather_counts = defaultdict(int)
    for row in building_rows:
        building_counts[row["split"]] += 1
    for row in weather_rows:
        weather_counts[row["split"]] += 1
    expected = {"train": 2100, "val": 450, "test": 450}
    if dict(building_counts) != expected or dict(weather_counts) != expected:
        raise ValueError(
            f"Unexpected split sizes: building={dict(building_counts)}, weather={dict(weather_counts)}"
        )

    _assert_disjoint(_values_by_split(building_rows, "building_id"), "building_id")
    _assert_disjoint(_values_by_split(weather_rows, "weather_id"), "weather_id")

    building_subsets = _values_by_split(building_rows, "subset")
    if building_subsets != {"train": {"P"}, "val": {"P"}, "test": {"M"}}:
        raise ValueError(f"Unexpected building-OOD domains: {building_subsets}")
    if set().union(*_values_by_split(building_rows, "weather_id").values()) != {"Chicago"}:
        raise ValueError("Building-OOD must hold weather fixed to Chicago")

    weather_buildings = _values_by_split(weather_rows, "building_id")
    if not (
        weather_buildings["train"]
        == weather_buildings["val"]
        == weather_buildings["test"]
    ):
        raise ValueError("Weather-OOD partitions must use identical building sets")
    if len(weather_buildings["train"]) != 150:
        raise ValueError("Weather-OOD must contain exactly 150 buildings")

    weather_ids = _values_by_split(weather_rows, "weather_id")
    expected_weather_ids = {
        split_name: set(values) for split_name, values in WEATHER_PARTITIONS.items()
    }
    if weather_ids != expected_weather_ids:
        raise ValueError(f"Unexpected Weather-OOD locations: {weather_ids}")

    subset_by_building = {
        row["building_id"]: row["subset"] for row in weather_rows
    }
    subset_counts = defaultdict(int)
    for building_id in weather_buildings["train"]:
        subset_counts[subset_by_building[building_id]] += 1
    if dict(subset_counts) != {"P": 75, "M": 75}:
        raise ValueError(f"Unexpected Weather-OOD building domains: {dict(subset_counts)}")


def _metadata(
    building_rows: list[dict[str, str]], weather_rows: list[dict[str, str]]
) -> dict:
    weather_buildings = _values_by_split(weather_rows, "building_id")["train"]
    weather_subsets = defaultdict(int)
    for building_id in weather_buildings:
        row = next(row for row in weather_rows if row["building_id"] == building_id)
        weather_subsets[row["subset"]] += 1

    return {
        "schema_version": 1,
        "standard_splits": {
            "split_p.csv": {
                "protocol": "iid-case-level",
                "note": "Building IDs and weather IDs may overlap across train, validation, and test.",
            },
            "split_m.csv": {
                "protocol": "iid-case-level",
                "note": "Building IDs and weather IDs may overlap across train, validation, and test.",
            },
        },
        "building_ood": {
            "file": "split_building_bias.csv",
            "scenario": "building-bias",
            "protocol": "building-ood-p-to-m",
            "weather_ids": ["Chicago"],
            "cases": {"train": 2100, "val": 450, "test": 450},
            "domains": {"train": "P", "val": "P", "test": "M"},
            "building_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        },
        "weather_location_ood": {
            "file": "split_weather_bias.csv",
            "scenario": "weather-bias",
            "protocol": "weather-location-ood",
            "cases": {"train": 2100, "val": 450, "test": 450},
            "buildings": {
                "total": len(weather_buildings),
                "P": weather_subsets["P"],
                "M": weather_subsets["M"],
                "shared_across_partitions": True,
                "ids": sorted(weather_buildings, key=int),
            },
            "weather_ids": WEATHER_PARTITIONS,
            "weather_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "note": "Partitions are disjoint by weather location ID; this is not an ASHRAE climate-zone-disjoint protocol.",
        },
    }


def _write_or_check(path: Path, expected: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == expected:
        print(f"OK {path}")
        return True
    if check:
        print(f"STALE {path}")
        return False
    path.write_text(expected, encoding="utf-8")
    print(f"WROTE {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate ArchEGraph OOD evaluation splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing manifest.csv and split/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check generated content without writing files.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_dir = data_root / "split"
    building_path = split_dir / "split_building_bias.csv"
    weather_path = split_dir / "split_weather_bias.csv"
    metadata_path = split_dir / "split_metadata.json"

    subset_by_building = _subset_map(split_dir)
    manifest_by_sample, weather_by_building = _manifest_index(data_root / "manifest.csv")
    building_rows = _repair_building_split(
        _read_csv(building_path), subset_by_building, manifest_by_sample
    )
    selected_buildings = _selected_buildings(_read_csv(weather_path))
    weather_rows = _generate_weather_split(
        selected_buildings,
        subset_by_building,
        manifest_by_sample,
        weather_by_building,
    )
    _validate(building_rows, weather_rows)

    metadata = json.dumps(_metadata(building_rows, weather_rows), indent=2) + "\n"
    results = [
        _write_or_check(building_path, _serialize_csv(building_rows), args.check),
        _write_or_check(weather_path, _serialize_csv(weather_rows), args.check),
        _write_or_check(metadata_path, metadata, args.check),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())