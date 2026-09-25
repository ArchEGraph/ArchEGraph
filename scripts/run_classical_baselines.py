#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph2energy.classical_baselines import (  # noqa: E402
    BASELINE_NAMES,
    RegressionMetrics,
    aggregate_zone_geometry,
    build_estimator,
    build_feature_rows,
    normalize_weather,
    prepare_dynamic_features,
    sample_case_rows,
)
from graph2energy.prep.dataload import (  # noqa: E402
    SignLogZScoreScaler,
    build_case_splits_from_csv,
    read_pack_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train non-graph degree-hour, geometry-tree, and RC-inspired "
            "Graph2Energy baselines."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR.parent / "data" / "ArchEGraph-demo",
        help="ArchEGraph PACK root containing manifest.csv.",
    )
    parser.add_argument(
        "--split",
        default="split_demo",
        help="Split CSV path or a name resolved under DATA_DIR/split.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(BASELINE_NAMES),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degree-base-temperature", type=float, default=18.0)
    parser.add_argument("--energy-alpha", type=float, default=80.0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=250_000,
        help="Maximum sampled zone-hour rows used to fit each model; 0 uses all rows.",
    )
    parser.add_argument(
        "--max-eval-samples-per-case",
        type=int,
        default=0,
        help="Optional evaluation cap per case; 0 evaluates every zone-hour.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=100_000)
    return parser.parse_args()


def resolve_split_path(data_dir: Path, split: str) -> Path:
    supplied = Path(split)
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend(
            [
                data_dir / "split" / supplied,
                data_dir / "split" / f"{supplied}.csv",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve split CSV from {split!r}.")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_manifest_index(data_dir: Path) -> dict[str, dict[str, str]]:
    manifest = read_pack_manifest(data_dir)
    return {
        str(row["sample_id"]): {
            "building_id": str(row["building_id"]),
            "building_file": f"{str(row['sample_id']).split('__', 1)[0]}.npz",
            "weather_id": str(row["weather_id"]),
            "energy_file": str(row["energy_file"]),
            "n_steps": int(row["n_steps"]),
            "n_spaces": int(row["n_spaces"]),
        }
        for _, row in manifest.iterrows()
    }


def load_case(data_dir: Path, row: dict[str, str]):
    building_path = data_dir / "building" / row["building_file"]
    if not building_path.is_file():
        building_path = data_dir / "building" / f"{row['building_id']}.npz"
    building = load_npz(building_path)
    weather_npz = load_npz(data_dir / "weather" / f"{row['weather_id']}.npz")
    energy_npz = load_npz(data_dir / "energy" / row["energy_file"])

    weather = normalize_weather(weather_npz["values"], weather_npz["columns"])
    geometry = aggregate_zone_geometry(building)
    energy = np.asarray(energy_npz["values"], dtype=np.float64)
    energy_columns = [str(value) for value in np.asarray(energy_npz["columns"]).tolist()]

    if "valid_energy_spaces" in building:
        valid_spaces = [
            str(value) for value in np.asarray(building["valid_energy_spaces"]).tolist()
        ]
        column_lookup = {name: index for index, name in enumerate(energy_columns)}
        matched_indices = [column_lookup[name] for name in valid_spaces if name in column_lookup]
        if matched_indices:
            energy = energy[:, matched_indices]

    if energy.shape[1] > len(geometry):
        energy = energy[:, : len(geometry)]
    elif energy.shape[1] < len(geometry):
        padding = np.zeros(
            (energy.shape[0], len(geometry) - energy.shape[1]), dtype=energy.dtype
        )
        energy = np.column_stack([energy, padding])

    usable_steps = min(len(weather.values), len(energy))
    if usable_steps == 0:
        raise ValueError("Case has no aligned weather and energy time steps.")
    return weather.__class__(weather.values[:usable_steps], weather.columns), geometry, energy[
        :usable_steps
    ]


def validate_case_ids(
    case_splits: dict[str, list[str]], manifest_index: dict[str, dict[str, str]]
) -> None:
    missing = sorted(
        {
            case_id
            for case_ids in case_splits.values()
            for case_id in case_ids
            if case_id not in manifest_index
        }
    )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Split references {len(missing)} cases absent from manifest.csv: {preview}"
        )


def fit_energy_scaler(
    data_dir: Path,
    manifest_index: dict[str, dict[str, str]],
    train_case_ids: list[str],
    energy_alpha: float,
) -> SignLogZScoreScaler:
    scaler = SignLogZScoreScaler(alpha=energy_alpha)
    for index, case_id in enumerate(train_case_ids, start=1):
        _, _, energy = load_case(data_dir, manifest_index[case_id])
        scaler.partial_fit(energy)
        print(f"\rFit target scaler: {index}/{len(train_case_ids)}", end="", flush=True)
    print()
    if scaler.mean_ is None:
        raise ValueError("No finite training targets were found.")
    return scaler


def allocate_case_samples(total_samples: int, case_sizes: list[int]) -> list[int]:
    n_cases = len(case_sizes)
    if n_cases == 0:
        raise ValueError("Cannot allocate samples without training cases.")
    if total_samples == 0 or total_samples >= sum(case_sizes):
        return list(case_sizes)
    if total_samples < n_cases:
        raise ValueError(
            f"max_train_samples ({total_samples}) must be at least the number of "
            f"training cases ({n_cases})."
        )

    allocations = [0] * n_cases
    active = set(range(n_cases))
    remaining = int(total_samples)
    while active:
        base, remainder = divmod(remaining, len(active))
        completed = []
        for position, index in enumerate(sorted(active)):
            requested = base + (position < remainder)
            capacity = case_sizes[index] - allocations[index]
            assigned = min(requested, capacity)
            allocations[index] += assigned
            remaining -= assigned
            if allocations[index] == case_sizes[index]:
                completed.append(index)
        active.difference_update(completed)
        if remaining == 0:
            break
        if not completed:
            raise RuntimeError("Failed to distribute the requested training samples.")
    return allocations


def collect_training_rows(
    baseline_name: str,
    data_dir: Path,
    manifest_index: dict[str, dict[str, str]],
    train_case_ids: list[str],
    energy_scaler: SignLogZScoreScaler,
    max_train_samples: int,
    degree_base_temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    case_sizes = [
        manifest_index[case_id]["n_steps"] * manifest_index[case_id]["n_spaces"]
        for case_id in train_case_ids
    ]
    case_sample_counts = allocate_case_samples(max_train_samples, case_sizes)
    rng = np.random.default_rng(seed)
    total_samples = sum(case_sample_counts)
    all_features = None
    all_targets = np.empty(total_samples, dtype=np.float32)
    offset = 0

    for index, (case_id, sample_count) in enumerate(
        zip(train_case_ids, case_sample_counts), start=1
    ):
        weather, geometry, energy = load_case(data_dir, manifest_index[case_id])
        dynamic = prepare_dynamic_features(weather, degree_base_temperature)
        time_indices, zone_indices = sample_case_rows(
            len(energy), len(geometry), sample_count, rng
        )
        features = build_feature_rows(
            baseline_name, dynamic, geometry, time_indices, zone_indices
        )
        targets = energy_scaler.transform(energy[time_indices, zone_indices])
        if all_features is None:
            all_features = np.empty(
                (total_samples, features.shape[1]), dtype=features.dtype
            )
        end = offset + len(features)
        all_features[offset:end] = features
        all_targets[offset:end] = np.asarray(targets, dtype=np.float32)
        offset = end
        print(
            f"\rCollect {baseline_name}: {index}/{len(train_case_ids)} cases",
            end="",
            flush=True,
        )
    print()
    if all_features is None or offset != total_samples:
        raise RuntimeError(
            f"Collected {offset} training rows, expected {total_samples}."
        )
    return all_features, all_targets


def iter_case_indices(
    n_time_steps: int,
    n_zones: int,
    batch_size: int,
    max_samples: int,
    rng: np.random.Generator,
):
    total_rows = n_time_steps * n_zones
    if max_samples > 0 and max_samples < total_rows:
        time_indices, zone_indices = sample_case_rows(
            n_time_steps, n_zones, max_samples, rng
        )
        for start in range(0, len(time_indices), batch_size):
            end = start + batch_size
            yield time_indices[start:end], zone_indices[start:end]
        return

    for start in range(0, total_rows, batch_size):
        flat_indices = np.arange(start, min(start + batch_size, total_rows))
        yield np.divmod(flat_indices, n_zones)


def evaluate_model(
    baseline_name: str,
    model,
    data_dir: Path,
    manifest_index: dict[str, dict[str, str]],
    case_ids: list[str],
    energy_scaler: SignLogZScoreScaler,
    degree_base_temperature: float,
    prediction_batch_size: int,
    max_samples_per_case: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    normalized_metrics = RegressionMetrics()
    original_metrics = RegressionMetrics()
    rng = np.random.default_rng(seed)

    for index, case_id in enumerate(case_ids, start=1):
        weather, geometry, energy = load_case(data_dir, manifest_index[case_id])
        dynamic = prepare_dynamic_features(weather, degree_base_temperature)
        for time_indices, zone_indices in iter_case_indices(
            len(energy),
            len(geometry),
            prediction_batch_size,
            max_samples_per_case,
            rng,
        ):
            features = build_feature_rows(
                baseline_name, dynamic, geometry, time_indices, zone_indices
            )
            targets = energy[time_indices, zone_indices]
            normalized_targets = energy_scaler.transform(targets)
            normalized_predictions = model.predict(features)
            predictions = energy_scaler.inverse_transform(normalized_predictions)
            normalized_metrics.update(normalized_targets, normalized_predictions)
            original_metrics.update(targets, predictions)
        print(
            f"\rEvaluate {baseline_name}: {index}/{len(case_ids)} cases",
            end="",
            flush=True,
        )
    print()
    return {
        "normalized": normalized_metrics.compute(),
        "original": original_metrics.compute(),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    split_path = resolve_split_path(data_dir, args.split)
    case_splits = build_case_splits_from_csv(split_path)
    manifest_index = build_manifest_index(data_dir)
    validate_case_ids(case_splits, manifest_index)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT_DIR / "cache" / "classical_baselines" / split_path.stem
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(data_dir),
        "split_csv": str(split_path),
        "models": list(args.models),
        "seed": args.seed,
        "degree_base_temperature": args.degree_base_temperature,
        "energy_alpha": args.energy_alpha,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples_per_case": args.max_eval_samples_per_case,
        "prediction_batch_size": args.prediction_batch_size,
        "case_counts": {name: len(values) for name, values in case_splits.items()},
    }
    write_json(output_dir / "config.json", config)

    energy_scaler = fit_energy_scaler(
        data_dir,
        manifest_index,
        case_splits["train"],
        args.energy_alpha,
    )
    summary_rows = []
    for model_index, baseline_name in enumerate(args.models):
        model_dir = output_dir / baseline_name
        model_dir.mkdir(parents=True, exist_ok=True)
        features, targets = collect_training_rows(
            baseline_name,
            data_dir,
            manifest_index,
            case_splits["train"],
            energy_scaler,
            args.max_train_samples,
            args.degree_base_temperature,
            args.seed + model_index,
        )
        model = build_estimator(baseline_name, args.seed)
        print(f"Fit {baseline_name}: {features.shape[0]} rows x {features.shape[1]} features")
        model.fit(features, targets)
        del features, targets

        metrics = {
            split_name: evaluate_model(
                baseline_name,
                model,
                data_dir,
                manifest_index,
                case_splits[split_name],
                energy_scaler,
                args.degree_base_temperature,
                args.prediction_batch_size,
                args.max_eval_samples_per_case,
                args.seed + model_index + split_index * 10_000,
            )
            for split_index, split_name in enumerate(("val", "test"), start=1)
        }
        model_config = {
            **config,
            "model": baseline_name,
            "feature_count": int(model.n_features_in_),
            "target_scaler": {
                "type": "SignLogZScoreScaler",
                "alpha": energy_scaler.alpha,
                "mean": energy_scaler.mean_,
                "std": energy_scaler.std_,
            },
        }
        write_json(model_dir / "config.json", model_config)
        write_json(model_dir / "metrics.json", metrics)
        joblib.dump(model, model_dir / "model.joblib")

        for split_name, scales in metrics.items():
            summary_rows.append(
                {
                    "model": baseline_name,
                    "split": split_name,
                    **{
                        f"{scale}_{metric}": value
                        for scale, values in scales.items()
                        for metric, value in values.items()
                    },
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    print(f"Saved baseline artifacts to {output_dir}")


if __name__ == "__main__":
    main()
