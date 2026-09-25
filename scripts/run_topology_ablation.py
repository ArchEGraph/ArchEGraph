#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT_DIR / "main.py"
CACHE_ROOT = ROOT_DIR / "cache"
TOPOLOGY_MODES = ("original", "shuffled")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired G2E experiments with original and degree-preserving "
            "shuffled face-to-space topology."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "graph2energy.minimal.json",
    )
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Training seeds. Use at least three seeds for reported results.",
    )
    parser.add_argument(
        "--topology_seed",
        type=int,
        default=42,
        help="Fixed shuffled-topology seed shared by all training seeds.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Prefix for run directories and the summary directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without starting training.",
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value]
    normalized = "".join(safe).strip("._")
    if not normalized:
        raise ValueError("experiment_name must contain at least one alphanumeric character.")
    return normalized


def _resolve_split_argument(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _build_command(
    args: argparse.Namespace,
    mode: str,
    train_seed: int,
    run_name: str,
) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_PATH),
        "--task",
        "graph2energy",
        "--config",
        str(args.config.expanduser().resolve()),
        "--device",
        args.device,
        "--seed",
        str(train_seed),
        "--topology_mode",
        mode,
        "--topology_seed",
        str(args.topology_seed),
        "--run_name",
        run_name,
    ]

    if args.data_dir is not None:
        command.extend(["--data_dir", str(args.data_dir.expanduser().resolve())])
    split = _resolve_split_argument(args.split)
    if split is not None:
        command.extend(["--split", split])
    if args.model is not None:
        command.extend(["--graph_model", args.model])
    if args.epochs is not None:
        command.extend(["--graph_epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--graph_batch_size", str(args.batch_size)])
    return command


def _flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "best_val_rmse": payload.get("best_val_rmse"),
    }
    for scope in ("test", "test_original"):
        values = payload.get(scope)
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            if isinstance(value, (int, float)):
                row[f"{scope}.{metric}"] = value
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    excluded = {"mode", "train_seed", "topology_seed", "run_name"}
    for mode in TOPOLOGY_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        metric_names = sorted(
            {
                key
                for row in mode_rows
                for key, value in row.items()
                if key not in excluded and isinstance(value, (int, float))
            }
        )
        result[mode] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in mode_rows if metric in row]
            result[mode][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return result


def _paired_deltas(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_seed = {
        (int(row["train_seed"]), str(row["mode"])): row
        for row in rows
    }
    deltas: dict[str, list[float]] = {}
    for seed in sorted({int(row["train_seed"]) for row in rows}):
        original = by_seed.get((seed, "original"))
        shuffled = by_seed.get((seed, "shuffled"))
        if original is None or shuffled is None:
            continue
        for metric in sorted(set(original) & set(shuffled)):
            if metric in {"mode", "train_seed", "topology_seed", "run_name"}:
                continue
            if isinstance(original[metric], (int, float)) and isinstance(
                shuffled[metric], (int, float)
            ):
                deltas.setdefault(metric, []).append(
                    float(shuffled[metric]) - float(original[metric])
                )

    return {
        metric: {
            "mean_shuffled_minus_original": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
        for metric, values in deltas.items()
    }


def _write_summary(
    summary_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "conditions": list(TOPOLOGY_MODES),
            "shuffled_topology": (
                "Within-building permutation of face-to-space destinations; "
                "preserves edge count, edge attributes, source faces, and per-space degree."
            ),
            "training_seeds": args.seeds,
            "topology_seed": args.topology_seed,
            "config": str(args.config.expanduser().resolve()),
            "data_dir": str(args.data_dir.expanduser().resolve()) if args.data_dir else None,
            "split": args.split,
            "model": args.model,
        },
        "runs": rows,
        "aggregate": _aggregate(rows),
        "paired_delta": _paired_deltas(rows),
    }
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = sorted({key for row in rows for key in row})
    with (summary_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    experiment_name = _safe_name(
        args.experiment_name
        or f"topology_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    rows: list[dict[str, Any]] = []

    for train_seed in args.seeds:
        for mode in TOPOLOGY_MODES:
            run_name = f"{experiment_name}_{mode}_train{train_seed}_topo{args.topology_seed}"
            command = _build_command(args, mode, train_seed, run_name)
            print(f"[{mode} | train_seed={train_seed}] {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue

            subprocess.run(command, cwd=ROOT_DIR, check=True)
            metrics_path = CACHE_ROOT / run_name / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Training completed without metrics: {metrics_path}")
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            rows.append(
                {
                    "mode": mode,
                    "train_seed": train_seed,
                    "topology_seed": args.topology_seed,
                    "run_name": run_name,
                    **_flatten_metrics(metrics),
                }
            )

    if args.dry_run:
        return 0

    summary_dir = CACHE_ROOT / experiment_name
    _write_summary(summary_dir, args, rows)
    print(f"Summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())