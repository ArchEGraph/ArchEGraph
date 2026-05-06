from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from model import build_model
except Exception:
    from models.model import build_model
from prep.dataload import GraphToTopologyDataset, build_sample_splits_from_csv, ids_from_building
from utils.metric import run_epoch


def _normalize_run_name(run_name) -> str | None:
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


def _make_json_serializable(value):
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _build_run_config_payload(
    *,
    run_name: str,
    task: str,
    cache_root: Path,
    run_dir: Path,
    data_dir: Path,
    split_csv_path: Path,
    device: torch.device,
    seed: int,
    deterministic: bool,
    model_name: str,
    raw_config,
    task_config,
):
    return _make_json_serializable(
        {
            "schema_version": 1,
            "task": task,
            "run_name": run_name,
            "model": model_name,
            "runtime": {
                "device": str(device),
                "seed": seed,
                "deterministic": deterministic,
            },
            "paths": {
                "cache_root": cache_root,
                "run_dir": run_dir,
                "data_dir": data_dir,
                "split_csv": split_csv_path,
            },
            "task_config": task_config,
            "raw_config": raw_config,
        }
    )


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


def collate_list(batch):
    return batch


def train_Mesh2Graph(config):
    scheduler_patience = 3
    scheduler_factor = 0.5
    scheduler_min_lr = 1e-7

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    _configure_reproducibility(seed=seed, deterministic=deterministic)

    model_name = str(config.get("model", "TopoTransformer"))
    epochs = int(config.get("epochs", 30))
    batch_size = int(config.get("batch_size", 8))
    lr = float(config.get("lr", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    hidden_dim = int(config.get("hidden_dim", 256))
    max_spaces = int(config.get("max_spaces", 64))
    metric_threshold = float(config.get("metric_threshold", 0.5))
    early_stop_patience = int(config.get("early_stop_patience", 8))

    device_cfg = config.get("device", "cpu")
    device = device_cfg if isinstance(device_cfg, torch.device) else torch.device(device_cfg)

    data_dir_cfg = config.get("data_dir", "/mnt/z/lyh/SRT/PACK")
    split_csv_cfg = config.get("split_csv", None)

    run_name = _normalize_run_name(config.get("run_name")) or f"M2G_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cache_root = Path(config["cache_root"]).expanduser().resolve()
    save_dir = cache_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir_cfg)
    building_root = data_dir / "building"
    geometry_root = data_dir / "geometry"
    if not building_root.exists():
        raise RuntimeError(f"Building folder not found: {building_root}")
    if not geometry_root.exists():
        raise RuntimeError(f"Geometry folder not found: {geometry_root}")

    all_ids = ids_from_building(building_root)
    if len(all_ids) == 0:
        raise RuntimeError(f"No building npz files in {building_root}")

    if split_csv_cfg is None or str(split_csv_cfg).strip() == "":
        raise RuntimeError("Missing required config: split_csv")

    split_csv_path = Path(split_csv_cfg)
    split_dict = build_sample_splits_from_csv(split_csv_path, allowed_ids=all_ids)
    train_ids = split_dict["train"]
    val_ids = split_dict["val"]
    test_ids = split_dict["test"]

    train_ds = GraphToTopologyDataset(building_root, geometry_root, train_ids, max_spaces=max_spaces)
    feat_dim = train_ds.face_feat_dim
    val_ds = GraphToTopologyDataset(building_root, geometry_root, val_ids, max_spaces=max_spaces, face_feat_dim=feat_dim)
    test_ds = GraphToTopologyDataset(building_root, geometry_root, test_ids, max_spaces=max_spaces, face_feat_dim=feat_dim)

    loader_kwargs = {
        "pin_memory": device.type == "cuda",
    }

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collate_list,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_list, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_list, **loader_kwargs)

    model = build_model(model_name, face_in_dim=feat_dim, hidden_dim=hidden_dim, max_spaces=max_spaces).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    best_val_sf = -1.0
    no_improve = 0
    best_state = None

    print(
        f"Using {len(all_ids)} candidate buildings, "
        f"split_csv={split_csv_path}, "
        f"train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)}, "
        f"face_feat_dim={feat_dim}, max_spaces={max_spaces}, model_name={model_name}, "
        f"seed={seed}, deterministic={deterministic}"
    )

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optim, device, train=True, metric_thr=metric_threshold)
        val_m = run_epoch(model, val_loader, optimizer=None, device=device, train=False, metric_thr=metric_threshold)
        scheduler.step(val_m["sf_f1"])
        current_lr = float(optim.param_groups[0]["lr"])

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_m['loss']:.4f}, train_ff_f1={train_m['ff_f1']:.4f}, "
            f"train_s_f1={train_m['s_f1']:.4f}, train_sf_f1={train_m['sf_f1']:.4f} | "
            f"val_loss={val_m['loss']:.4f}, val_ff_f1={val_m['ff_f1']:.4f}, "
            f"val_s_f1={val_m['s_f1']:.4f}, val_sf_f1={val_m['sf_f1']:.4f}, lr={current_lr:.2e}"
        )

        if val_m["sf_f1"] > best_val_sf:
            best_val_sf = val_m["sf_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"[EarlyStop] no val sf_f1 improvement for {no_improve} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = run_epoch(model, test_loader, optimizer=None, device=device, train=False, metric_thr=metric_threshold)
    print(
        f"[Test] loss={test_m['loss']:.4f}, ff_f1={test_m['ff_f1']:.4f}, "
        f"s_f1={test_m['s_f1']:.4f}, sf_f1={test_m['sf_f1']:.4f}"
    )

    ckpt = save_dir / "model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_name": run_name,
            "config": {
                "model_name": model_name,
                "face_in_dim": feat_dim,
                "hidden_dim": hidden_dim,
                "max_spaces": max_spaces,
                "split_csv": str(split_csv_path),
            },
            "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        },
        ckpt,
    )

    metrics_path = save_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        payload = {
            "run_name": run_name,
            "model": model_name,
            "best_val_sf_f1": best_val_sf,
            "test": test_m,
        }
        json.dump(payload, f, indent=2)

    config_payload = _build_run_config_payload(
        run_name=run_name,
        task="mesh2graph",
        cache_root=cache_root,
        run_dir=save_dir,
        data_dir=data_dir,
        split_csv_path=split_csv_path,
        device=device,
        seed=seed,
        deterministic=deterministic,
        model_name=model_name,
        raw_config=config,
        task_config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "max_spaces": max_spaces,
            "metric_threshold": metric_threshold,
            "early_stop_patience": early_stop_patience,
        },
    )
    config_path = save_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
