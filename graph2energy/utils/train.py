import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from prep.dataload import (
    GraphDataset,
    build_multiscale_time_features,
    fit_global_scalers,
    load_case_data,
    _is_preprocessed_pyg_case,
)
from model import build_model, is_day_model
from utils.metric import batch_mse_loss, evaluate_predictions, flatten_numpy


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
    data_dir: Path | None,
    split_csv_path: Path | None,
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


def _dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _resolve_explicit_split_cases(data_dict, config):
    explicit = config.get("case_splits")
    if not explicit:
        raise ValueError("Missing config['case_splits'].")

    available = set(data_dict.keys())
    train = [k for k in _dedupe_keep_order([str(v) for v in explicit.get("train", [])]) if k in available]
    val = [k for k in _dedupe_keep_order([str(v) for v in explicit.get("val", [])]) if k in available]
    test = [k for k in _dedupe_keep_order([str(v) for v in explicit.get("test", [])]) if k in available]

    if not train or not val or not test:
        raise ValueError("Invalid case_splits: train/val/test must all be non-empty after filtering.")

    return {"train": train, "val": val, "test": test}


def _infer_weather_input_dim(sample_case, use_multiscale_time_encoding, weather_encoding_strategy):
    if _is_preprocessed_pyg_case(sample_case):
        return int(sample_case["weather"].shape[1])

    sample_weather_df = build_multiscale_time_features(
        sample_case["weather"],
        use_multiscale_time_encoding=use_multiscale_time_encoding,
        drop_original_time_feature=True,
        weather_encoding_strategy=weather_encoding_strategy,
    )
    return int(sample_weather_df.shape[1])


def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True)
            pred = model(data)
            target = data.energy.to(device)

            _, batch_loss_sum, batch_samples = batch_mse_loss(pred, target)
            total_loss += batch_loss_sum
            total_samples += batch_samples

            pred_np, target_np = flatten_numpy(pred, target)
            all_preds.append(pred_np)
            all_targets.append(target_np)

    avg_loss = total_loss / max(total_samples, 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = evaluate_predictions(targets, preds)
    return avg_loss, metrics, preds, targets


def train_ArcheEnergy(data_dict, config):
    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    _configure_reproducibility(seed=seed, deterministic=deterministic)

    epochs = int(config.get("epochs", 100))
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    batch_size = int(config.get("batch_size", 1))
    patience = int(config.get("patience", 10))
    model_name = str(config.get("model", "F2SAttr"))

    device_cfg = config.get("device", "cpu")
    device = device_cfg if isinstance(device_cfg, torch.device) else torch.device(device_cfg)

    normalize_weather = bool(config.get("normalize_weather", True))
    normalize_energy = bool(config.get("normalize_energy", True))
    energy_scaler_alpha = float(config.get("energy_scaler_alpha", 80.0))
    use_multiscale_time_encoding = bool(config.get("use_multiscale_time_encoding", True))
    weather_encoding_strategy = str(config.get("weather_encoding_strategy", "base"))

    split_cases = _resolve_explicit_split_cases(data_dict, config)
    sample_case = load_case_data(data_dict[next(iter(data_dict.keys()))])
    weather_input_dim = _infer_weather_input_dim(
        sample_case,
        use_multiscale_time_encoding=use_multiscale_time_encoding,
        weather_encoding_strategy=weather_encoding_strategy,
    )

    weather_scaler = None
    energy_scaler = None
    data_dir_cfg = config.get("data_dir")
    split_csv_cfg = config.get("split_csv")
    if normalize_weather or normalize_energy:
        weather_scaler, energy_scaler = fit_global_scalers(
            data_dict,
            split_cases["train"],
            energy_alpha=energy_scaler_alpha,
            use_multiscale_time_encoding=use_multiscale_time_encoding,
            drop_original_time_feature=True,
            weather_encoding_strategy=weather_encoding_strategy,
        )

    is_day = is_day_model(model_name)
    day_length = 8760 if is_day else 24
    day_stride = 8760 if is_day else 24
    granularity = "day" if is_day else "hour"

    common_dataset_kwargs = {
        "split_ratio": (0.7, 0.15, 0.15),
        "seed": seed,
        "weather_scaler": weather_scaler if normalize_weather else None,
        "energy_scaler": energy_scaler if normalize_energy else None,
        "energy_alpha": energy_scaler_alpha,
        "normalize_weather": normalize_weather,
        "normalize_energy": normalize_energy,
        "use_multiscale_time_encoding": use_multiscale_time_encoding,
        "drop_original_time_feature": True,
        "weather_encoding_strategy": weather_encoding_strategy,
        "granularity": granularity,
        "day_length": day_length,
        "day_stride": day_stride,
    }

    train_dataset = GraphDataset(
        data_dict,
        split="train",
        shuffle=True,
        case_keys=split_cases["train"],
        **common_dataset_kwargs,
    )
    val_dataset = GraphDataset(
        data_dict,
        split="val",
        shuffle=False,
        case_keys=split_cases["val"],
        **common_dataset_kwargs,
    )
    test_dataset = GraphDataset(
        data_dict,
        split="test",
        shuffle=False,
        case_keys=split_cases["test"],
        **common_dataset_kwargs,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
    }

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(model_name, weather_input_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_name = _normalize_run_name(config.get("run_name")) or f"G2E_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cache_root = Path(config["cache_root"]).expanduser().resolve()
    run_dir = cache_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reproducibility | seed={seed}, deterministic={deterministic}")

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for data in train_loader:
            data = data.to(device, non_blocking=True)
            pred = model(data)
            target = data.energy.to(device)
            loss, batch_loss_sum, batch_samples = batch_mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += batch_loss_sum
            train_samples += batch_samples

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss, val_metrics, _, _ = _evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_metrics['mae']:.4f}, "
            f"val_rmse={val_metrics['rmse']:.4f}, val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch:03d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, test_preds, test_targets = _evaluate(model, test_loader, device)
    print(
        f"Test | loss={test_loss:.4f}, mae={test_metrics['mae']:.4f}, "
        f"rmse={test_metrics['rmse']:.4f}, r2={test_metrics['r2']:.4f}"
    )

    payload = {
        "run_name": run_name,
        "model": model_name,
        "best_val_rmse": best_val_rmse,
        "test": test_metrics,
    }

    if normalize_energy and energy_scaler is not None:
        inv_preds = energy_scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(-1)
        inv_targets = energy_scaler.inverse_transform(test_targets.reshape(-1, 1)).reshape(-1)
        payload["test_original"] = evaluate_predictions(inv_targets, inv_preds)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    config_payload = _build_run_config_payload(
        run_name=run_name,
        task="graph2energy",
        cache_root=cache_root,
        run_dir=run_dir,
        data_dir=Path(data_dir_cfg).expanduser() if data_dir_cfg else None,
        split_csv_path=Path(split_csv_cfg).expanduser() if split_csv_cfg else None,
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
            "patience": patience,
            "normalize_weather": normalize_weather,
            "normalize_energy": normalize_energy,
            "energy_scaler_alpha": energy_scaler_alpha,
            "use_multiscale_time_encoding": use_multiscale_time_encoding,
            "weather_encoding_strategy": weather_encoding_strategy,
            "granularity": granularity,
            "day_length": day_length,
            "day_stride": day_stride,
            "case_splits": split_cases,
        },
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model_name,
            "weather_input_dim": weather_input_dim,
            "config": config,
        },
        run_dir / "model.pth",
    )
