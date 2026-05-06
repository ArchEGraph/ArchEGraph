from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


ARCH_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ARCH_ROOT.parent
DEFAULT_SPLIT_DIR = WORKSPACE_ROOT / "split"
DEFAULT_DATA_DIR = ARCH_ROOT / "data"
CACHE_ROOT = ARCH_ROOT / "cache"


def _is_pack_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    required = [
        path / "manifest.csv",
        path / "building",
        path / "geometry",
        path / "weather",
        path / "energy",
    ]
    return all(p.exists() for p in required)


def _candidate_pack_roots(path_hint: Path | None) -> list[Path]:
    candidates = []
    if path_hint is not None:
        candidates.append(path_hint)
        candidates.append(path_hint / "PACK")

    candidates.extend(
        [
            DEFAULT_DATA_DIR,
            DEFAULT_DATA_DIR / "PACK",  # legacy local-dir layout
            ARCH_ROOT / "data",
            ARCH_ROOT,
            WORKSPACE_ROOT / "data" / "PACK",
            WORKSPACE_ROOT / "data",
            WORKSPACE_ROOT,
        ]
    )

    # HF downloads may be placed under project subfolders; discover by manifest.csv.
    for search_root in [ARCH_ROOT, WORKSPACE_ROOT]:
        if not search_root.exists() or not search_root.is_dir():
            continue
        for manifest in search_root.rglob("manifest.csv"):
            candidates.append(manifest.parent)

    unique = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    return unique


def _resolve_pack_root(path_hint: Any, config_path: Path | None) -> Path:
    hint = _resolve_cfg_path(path_hint, config_path)
    hint_path = Path(hint).expanduser() if hint is not None else None

    for candidate in _candidate_pack_roots(hint_path):
        if _is_pack_root(candidate):
            return candidate

    msg = [
        "Unable to locate dataset root automatically.",
        "Expected a directory containing manifest.csv + building/geometry/weather/energy.",
        "Checked candidates:",
    ]
    msg.extend(f"- {str(p)}" for p in _candidate_pack_roots(hint_path))
    raise FileNotFoundError("\n".join(msg))


def _resolve_split_csv_path(
    task: str,
    split_selector: Any,
    pack_root: Path,
    config_path: Path | None,
) -> Path:
    default_split_name = "split_m.csv" if task == "mesh2graph" else "split_p.csv"

    requested_split_name = default_split_name
    raw_selector = "" if split_selector is None else str(split_selector).strip()
    if raw_selector:
        selector_name = Path(raw_selector).name
        selector_suffix = Path(selector_name).suffix.lower()
        if selector_suffix == ".csv":
            requested_split_name = selector_name
        elif "/" not in raw_selector and "\\" not in raw_selector:
            requested_split_name = f"{selector_name}.csv"

    explicit = _resolve_cfg_path(split_selector, config_path)
    explicit_path = Path(explicit).expanduser() if explicit is not None else None

    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    candidates.extend(
        [
            pack_root / requested_split_name,
            pack_root / "split" / requested_split_name,
            pack_root / "splits" / requested_split_name,
            pack_root.parent / requested_split_name,
            pack_root.parent / "split" / requested_split_name,
            pack_root.parent / "splits" / requested_split_name,
            ARCH_ROOT / "data" / requested_split_name,
            ARCH_ROOT / "data" / "split" / requested_split_name,
            ARCH_ROOT / "data" / "splits" / requested_split_name,
            DEFAULT_SPLIT_DIR / requested_split_name,
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    msg = [
        f"Unable to locate split CSV for task={task}.",
        f"Expected file name: {requested_split_name}",
        "Checked candidates:",
    ]
    msg.extend(f"- {str(p)}" for p in candidates)
    raise FileNotFoundError("\n".join(msg))


def _parse_args():
    parser = argparse.ArgumentParser(description="Unified ArchEGraph entry for mesh2graph and graph2energy.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path.")
    parser.add_argument("--task", type=str, default=None, choices=["mesh2graph", "graph2energy"], help="Task name.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable deterministic operations when possible (default from config or true).",
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Dataset root, or a parent folder containing the dataset root.",
    )
    parser.add_argument(
        "--split",
        "--split_name",
        dest="split",
        type=str,
        default=None,
        help="Split selector (e.g. split_m, split_p, split_m.csv, or a split CSV path).",
    )

    parser.add_argument("--mesh_model", type=str, default=None)
    # Legacy aliases kept for backward compatibility.
    parser.add_argument("--mesh_data_dir", type=Path, default=None)
    parser.add_argument("--mesh_split_csv", type=Path, default=None)
    parser.add_argument("--mesh_epochs", type=int, default=None)
    parser.add_argument("--mesh_batch_size", type=int, default=None)

    parser.add_argument("--graph_model", type=str, default=None)
    # Legacy aliases kept for backward compatibility.
    parser.add_argument("--graph_data_dir", type=Path, default=None)
    parser.add_argument("--graph_split_csv", type=Path, default=None)
    parser.add_argument("--graph_epochs", type=int, default=None)
    parser.add_argument("--graph_batch_size", type=int, default=None)

    return parser.parse_args()


def _load_json_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")
    return payload


def _resolve_cfg_path(value: Any, config_path: Path | None) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return value
    p = Path(text)
    if p.is_absolute() or config_path is None:
        return p
    return (config_path.parent / p).resolve()


def _normalize_run_name(run_name: Any) -> str | None:
    if run_name is None:
        return None
    text = str(run_name).strip()
    if text == "":
        return None

    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")

    normalized = "".join(safe).strip("._")
    return normalized or None


def _resolve_run_name(run_name: Any, task: str | None = None) -> str:
    normalized = _normalize_run_name(run_name)
    if normalized:
        return normalized
    task_prefix_map = {
        "mesh2graph": "M2G",
        "graph2energy": "G2E",
    }
    prefix = task_prefix_map.get(str(task).strip().lower(), "RUN")
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_device(device_name: str) -> torch.device:
    text = str(device_name).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(text)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _configure_reproducibility(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _task_config_from_json(payload: Dict[str, Any], task: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    section = payload.get(task)
    if isinstance(section, dict):
        cfg.update(section)

    for key, value in payload.items():
        if key in {"task", "mesh2graph", "graph2energy"}:
            continue
        if key not in cfg:
            cfg[key] = value
    return cfg


def _append_task_path(task_name: str):
    task_path = ARCH_ROOT / task_name
    if not task_path.exists():
        raise FileNotFoundError(f"Task folder not found: {task_path}")
    sys.path.insert(0, str(task_path))


def _run_mesh2graph(base_cfg: Dict[str, Any], args, device: torch.device, config_path: Path | None):
    _append_task_path("mesh2graph")
    from utils.train import train_Mesh2Graph

    data_hint = args.data_dir or args.mesh_data_dir or base_cfg.get("data_dir")
    split_selector = (
        args.split
        or args.mesh_split_csv
        or base_cfg.get("split")
        or base_cfg.get("split_name")
        or base_cfg.get("split_csv")
    )
    data_dir = _resolve_pack_root(data_hint, config_path)
    split_csv = _resolve_split_csv_path("mesh2graph", split_selector, data_dir, config_path)
    run_name = _resolve_run_name(
        args.run_name if args.run_name is not None else base_cfg.get("run_name"),
        task="mesh2graph",
    )

    cfg = {
        "device": device,
        "run_name": run_name,
        "cache_root": CACHE_ROOT.resolve(),
        "model": args.mesh_model or base_cfg.get("model", "TopoTransformer"),
        "data_dir": data_dir,
        "split_csv": split_csv,
        "epochs": args.mesh_epochs if args.mesh_epochs is not None else int(base_cfg.get("epochs", 30)),
        "lr": float(base_cfg.get("lr", 3e-4)),
        "batch_size": args.mesh_batch_size if args.mesh_batch_size is not None else int(base_cfg.get("batch_size", 8)),
        "seed": int(base_cfg.get("seed", 42)),
        "deterministic": _as_bool(base_cfg.get("deterministic"), True),
    }
    print(f"Running task=mesh2graph on device={device}")
    print(f"Resolved dataset | pack_root={data_dir} | split_csv={split_csv}")
    print(f"Resolved cache | root={CACHE_ROOT.resolve()} | run_name={run_name}")
    train_Mesh2Graph(config=cfg)


def _run_graph2energy(base_cfg: Dict[str, Any], args, device: torch.device, config_path: Path | None):
    _append_task_path("graph2energy")

    from prep.dataload import read_split_data
    from utils.train import train_ArcheEnergy

    run_name = _resolve_run_name(
        args.run_name if args.run_name is not None else base_cfg.get("run_name"),
        task="graph2energy",
    )
    model_name = args.graph_model or base_cfg.get("model", "F2SAttr")
    data_hint = args.data_dir or args.graph_data_dir or base_cfg.get("data_dir")
    split_selector = (
        args.split
        or args.graph_split_csv
        or base_cfg.get("split")
        or base_cfg.get("split_name")
        or base_cfg.get("split_csv")
    )
    data_dir = _resolve_pack_root(data_hint, config_path)
    split_csv = _resolve_split_csv_path("graph2energy", split_selector, data_dir, config_path)

    print(f"Running task=graph2energy on device={device}")
    print(f"Resolved dataset | pack_root={data_dir} | split_csv={split_csv}")
    data_dict, case_splits = read_split_data(
        pack_dir=data_dir,
        split_csv_path=split_csv,
        verbose=True,
    )

    if not data_dict:
        raise RuntimeError("No cases loaded from split CSV. Check split file and data_dir.")

    train_cfg = {
        "device": device,
        "model": model_name,
        "run_name": run_name,
        "cache_root": CACHE_ROOT.resolve(),
        "case_splits": case_splits,
        "seed": int(base_cfg.get("seed", 42)),
        "deterministic": _as_bool(base_cfg.get("deterministic"), True),
    }
    if args.graph_epochs is not None:
        train_cfg["epochs"] = int(args.graph_epochs)
    elif "epochs" in base_cfg:
        train_cfg["epochs"] = int(base_cfg["epochs"])

    if args.graph_batch_size is not None:
        train_cfg["batch_size"] = int(args.graph_batch_size)
    elif "batch_size" in base_cfg:
        train_cfg["batch_size"] = int(base_cfg["batch_size"])

    print(f"Resolved cache | root={CACHE_ROOT.resolve()} | run_name={run_name}")
    train_ArcheEnergy(data_dict, config=train_cfg)


def main():
    args = _parse_args()
    json_cfg = _load_json_config(args.config)

    task = args.task or str(json_cfg.get("task", "")).strip().lower()
    if task not in {"mesh2graph", "graph2energy"}:
        raise ValueError("Please set --task to mesh2graph or graph2energy, or provide task in --config JSON.")

    base_cfg = _task_config_from_json(json_cfg, task)
    device = _resolve_device(args.device)

    seed = int(args.seed if args.seed is not None else base_cfg.get("seed", json_cfg.get("seed", 42)))
    deterministic = _as_bool(
        args.deterministic if args.deterministic is not None else base_cfg.get("deterministic", json_cfg.get("deterministic", True)),
        True,
    )
    base_cfg["seed"] = seed
    base_cfg["deterministic"] = deterministic
    _configure_reproducibility(seed=seed, deterministic=deterministic)
    print(f"Reproducibility | seed={seed}, deterministic={deterministic}")

    if task == "mesh2graph":
        _run_mesh2graph(base_cfg, args, device, args.config)
    else:
        _run_graph2energy(base_cfg, args, device, args.config)


if __name__ == "__main__":
    main()
