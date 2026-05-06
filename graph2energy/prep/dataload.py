import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
import copy
import random
from typing import Optional
from typing import Dict, List, Sequence, Tuple, Union


def load_case_data(case_source):
    """Load a single case from in-memory dict or serialized file path."""
    if isinstance(case_source, dict):
        return case_source

    if isinstance(case_source, (str, Path)):
        case_path = Path(case_source)
        if case_path.suffix.lower() == ".pt":
            try:
                return torch.load(case_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Compatibility for older torch versions without weights_only.
                return torch.load(case_path, map_location="cpu")
        with open(case_path, "rb") as pf:
            return pickle.load(pf)

    raise TypeError(f"Unsupported case source type: {type(case_source)}")


def _is_preprocessed_pyg_case(case):
    return isinstance(case, dict) and bool(case.get("__pyg_preprocessed__"))


def build_multiscale_time_features(
    weather_df,
    use_multiscale_time_encoding=True,
    drop_original_time_feature=True,
    weather_encoding_strategy="base",
):
    """Return weather features with optional multi-scale cyclical time encoding."""
    if not isinstance(weather_df, pd.DataFrame):
        raise TypeError("weather_df must be a pandas DataFrame.")

    df = weather_df.copy()
    time_column_name = "time"
    day_period_hours = 24
    week_period_hours = 168
    time_col = None
    for col in df.columns:
        if str(col).strip().lower() == str(time_column_name).strip().lower():
            time_col = col
            break

    if use_multiscale_time_encoding:
        if time_col is not None:
            t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(t).any():
                t = np.arange(1, len(df) + 1, dtype=np.float64)
        else:
            t = np.arange(1, len(df) + 1, dtype=np.float64)

        # Use zero-based phase for cleaner daily/weekly periodic alignment.
        phase = t - 1.0
        day_angle = 2.0 * np.pi * phase / float(day_period_hours)
        week_angle = 2.0 * np.pi * phase / float(week_period_hours)

        df["time_day_sin"] = np.sin(day_angle)
        df["time_day_cos"] = np.cos(day_angle)
        df["time_week_sin"] = np.sin(week_angle)
        df["time_week_cos"] = np.cos(week_angle)

    strategy = str(weather_encoding_strategy).strip().lower()
    if strategy not in {"base", "delta", "context_3h_3d"}:
        raise ValueError(
            "Unsupported weather_encoding_strategy: "
            f"{weather_encoding_strategy}. Expected one of: base, delta, context_3h_3d"
        )

    feature_cols = [
        c for c in df.columns
        if c != time_col and not str(c).startswith("time_")
    ]

    if strategy == "delta":
        default_delta_cols = ["db", "rh", "ghr", "dnr", "dhr"]
        delta_fill_value = 0.0
        candidate_cols = [c for c in default_delta_cols if c in df.columns]

        for col in candidate_cols:
            col_values = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_d1"] = col_values.diff().fillna(delta_fill_value)

    elif strategy == "context_3h_3d":
        hour_offsets = [1, 2, 3]
        day_offsets = [24, 48, 72]

        for col in feature_cols:
            col_values = pd.to_numeric(df[col], errors="coerce")

            for hour in hour_offsets:
                df[f"{col}_lag_{hour}h"] = col_values.shift(hour).ffill().bfill().fillna(0.0)
                df[f"{col}_lead_{hour}h"] = col_values.shift(-hour).ffill().bfill().fillna(0.0)

            for day_hour in day_offsets:
                day = day_hour // 24
                df[f"{col}_lag_{day}d"] = col_values.shift(day_hour).ffill().bfill().fillna(0.0)
                df[f"{col}_lead_{day}d"] = col_values.shift(-day_hour).ffill().bfill().fillna(0.0)

    if time_col is not None and drop_original_time_feature:
        df = df.drop(columns=[time_col])

    return df


def _infer_weather_temperature_indices(weather_df):
    """Infer temperature-like weather column indices that should be converted to Kelvin."""
    if not isinstance(weather_df, pd.DataFrame):
        return []

    temp_col_names = {
        "db",
        "dry_bulb",
        "dry_bulb_temperature",
        "temperature",
        "temp",
    }
    indices = []
    for idx, col in enumerate(weather_df.columns):
        col_name = str(col).strip().lower()
        if col_name in temp_col_names:
            indices.append(idx)
    return indices


def _infer_weather_temperature_indices_from_columns(weather_columns):
    if weather_columns is None:
        return []

    temp_col_names = {
        "db",
        "dry_bulb",
        "dry_bulb_temperature",
        "temperature",
        "temp",
    }
    indices = []
    for idx, col in enumerate(weather_columns):
        if str(col).strip().lower() in temp_col_names:
            indices.append(idx)
    return indices


class KelvinAwareMinMaxScaler:
    """Convert temperature columns to Kelvin, then apply per-feature min-max scaling."""

    def __init__(self, temperature_indices=None, eps=1e-12, kelvin_offset=273.15):
        self.temperature_indices = list(temperature_indices or [])
        self.eps = eps
        self.kelvin_offset = kelvin_offset
        self.scaler = MinMaxScaler()

    def _to_kelvin(self, X):
        arr = np.asarray(X, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not self.temperature_indices:
            return arr
        valid_idx = [i for i in self.temperature_indices if 0 <= i < arr.shape[1]]
        if valid_idx:
            arr[:, valid_idx] = arr[:, valid_idx] + self.kelvin_offset
        return arr

    def _to_celsius(self, X):
        arr = np.asarray(X, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not self.temperature_indices:
            return arr
        valid_idx = [i for i in self.temperature_indices if 0 <= i < arr.shape[1]]
        if valid_idx:
            arr[:, valid_idx] = arr[:, valid_idx] - self.kelvin_offset
        return arr

    def fit(self, X):
        arr = self._to_kelvin(X)
        if arr.size == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        self.scaler.fit(arr)
        return self

    def partial_fit(self, X):
        arr = self._to_kelvin(X)
        if arr.size == 0:
            return self
        self.scaler.partial_fit(arr)
        return self

    def transform(self, X):
        arr = self._to_kelvin(X)
        return self.scaler.transform(arr)

    def inverse_transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        restored = self.scaler.inverse_transform(arr)
        return self._to_celsius(restored)

class SignLogZScoreScaler:
    """Sign-log transform followed by global z-score normalization."""

    def __init__(self, alpha=80.0, eps=1e-12):
        if alpha <= 0:
            raise ValueError("alpha must be positive for SignLogZScoreScaler.")
        self.alpha = float(alpha)
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def _sign_log_transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.sign(arr) * np.log1p(self.alpha * np.abs(arr))

    def _sign_log_inverse(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.sign(arr) * (np.expm1(np.abs(arr)) / self.alpha)

    def fit(self, X):
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self.mean_ = None
        self.std_ = None
        self.partial_fit(X)
        if self._count == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        return self

    def partial_fit(self, X):
        transformed = self._sign_log_transform(X)
        flat = transformed.reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return self

        self._count += int(finite.size)
        self._sum += float(finite.sum())
        self._sum_sq += float(np.square(finite).sum())

        mean = self._sum / float(self._count)
        var = max(self._sum_sq / float(self._count) - mean * mean, 0.0)
        self.mean_ = float(mean)
        self.std_ = float(np.sqrt(var))
        return self

    def transform(self, X):
        transformed = self._sign_log_transform(X)
        scale = max(self.std_, self.eps)
        return (transformed - self.mean_) / scale

    def inverse_transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        scale = max(self.std_, self.eps)
        restored = arr * scale + self.mean_
        return self._sign_log_inverse(restored)


def split_case_keys(data_dict, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """Create a deterministic case-level train/val/test split."""
    all_case_keys = list(data_dict.keys())
    rng = random.Random(seed)
    rng.shuffle(all_case_keys)

    n_cases = len(all_case_keys)
    n_train_cases = int(split_ratio[0] * n_cases)
    n_val_cases = int((split_ratio[0] + split_ratio[1]) * n_cases)

    return {
        "train": all_case_keys[:n_train_cases],
        "val": all_case_keys[n_train_cases:n_val_cases],
        "test": all_case_keys[n_val_cases:],
    }


def fit_global_scalers(
    data_dict,
    train_case_keys,
    energy_alpha=80.0,
    use_multiscale_time_encoding=True,
    drop_original_time_feature=True,
    weather_encoding_strategy="base",
):
    """Fit weather/energy scalers using only training cases."""
    if not train_case_keys:
        raise ValueError("Training split is empty. Cannot fit global scalers.")

    sample_case = load_case_data(data_dict[train_case_keys[0]])
    if _is_preprocessed_pyg_case(sample_case):
        weather_columns = sample_case.get("weather_columns")
        weather_temp_indices = sample_case.get("weather_temp_indices")
        if weather_temp_indices is None:
            weather_temp_indices = _infer_weather_temperature_indices_from_columns(weather_columns)
    else:
        sample_weather_df = build_multiscale_time_features(
            sample_case["weather"],
            use_multiscale_time_encoding=use_multiscale_time_encoding,
            drop_original_time_feature=drop_original_time_feature,
            weather_encoding_strategy=weather_encoding_strategy,
        )
        weather_temp_indices = _infer_weather_temperature_indices(sample_weather_df)

    scaler_weather = KelvinAwareMinMaxScaler(
        temperature_indices=weather_temp_indices,
    )
    scaler_energy = SignLogZScoreScaler(alpha=energy_alpha)

    for case_key in train_case_keys:
        case = load_case_data(data_dict[case_key])
        if _is_preprocessed_pyg_case(case):
            weather = case["weather"]
            energy = case["energy"]
            if not torch.is_tensor(weather):
                weather = torch.tensor(weather, dtype=torch.float)
            if not torch.is_tensor(energy):
                energy = torch.tensor(energy, dtype=torch.float)
            usable_steps = min(int(weather.shape[0]), int(energy.shape[0]))
            scaler_weather.partial_fit(weather[:usable_steps].cpu().numpy())
            scaler_energy.partial_fit(energy[:usable_steps].cpu().numpy())
        else:
            weather_df = build_multiscale_time_features(
                case["weather"],
                use_multiscale_time_encoding=use_multiscale_time_encoding,
                drop_original_time_feature=drop_original_time_feature,
                weather_encoding_strategy=weather_encoding_strategy,
            )
            graph = build_hetero_graph(case["building"])
            space_count = int(graph["space"].x.size(0))
            energy_df = _align_energy_df_with_graph(case, case["energy"], space_count)
            scaler_weather.partial_fit(weather_df.values)
            scaler_energy.partial_fit(energy_df.values)

    weather_seen = getattr(scaler_weather.scaler, "n_samples_seen_", 0)
    if scaler_energy.mean_ is None or int(weather_seen) <= 0:
        raise ValueError("No valid samples found when fitting global scalers.")
    return scaler_weather, scaler_energy


def _resolve_case_split(data_dict, split, split_ratio, seed, case_keys=None):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    split_cases = split_case_keys(data_dict, split_ratio=split_ratio, seed=seed)
    selected_cases = case_keys if case_keys is not None else split_cases[split]
    return selected_cases, split_cases


def _normalize_weather_tensor(weather, weather_df, normalize_weather, weather_scaler):
    if not normalize_weather:
        return weather, weather_scaler

    if weather_scaler is not None:
        scaler_weather = weather_scaler
    else:
        weather_temp_indices = []
        if weather_df is not None:
            weather_temp_indices = _infer_weather_temperature_indices(weather_df)
        scaler_weather = KelvinAwareMinMaxScaler(
            temperature_indices=weather_temp_indices,
        ).fit(weather.numpy())

    weather_np = scaler_weather.transform(weather.numpy())
    return torch.tensor(weather_np, dtype=torch.float), scaler_weather


def _normalize_energy_tensor(energy, normalize_energy, energy_scaler, energy_alpha):
    if not normalize_energy:
        return energy, energy_scaler

    if energy_scaler is not None:
        scaler_energy = energy_scaler
    else:
        scaler_energy = SignLogZScoreScaler(alpha=energy_alpha).fit(energy.numpy())

    energy_np = scaler_energy.transform(energy.numpy())
    return torch.tensor(energy_np, dtype=torch.float), scaler_energy


def _align_energy_df_with_graph(case, energy_df, space_count):
    """Align energy columns to graph space count for mixed-quality pack data."""
    aligned = energy_df

    valid_spaces = case.get("building", {}).get("valid_energy_spaces") if isinstance(case, dict) else None
    if valid_spaces and isinstance(aligned, pd.DataFrame):
        valid_spaces = [str(x) for x in valid_spaces]
        matched = [c for c in valid_spaces if c in aligned.columns]
        if matched:
            aligned = aligned.loc[:, matched]

    n_energy = int(aligned.shape[1])
    if n_energy == space_count:
        return aligned

    if n_energy > space_count:
        return aligned.iloc[:, :space_count]

    # Rare case: fewer energy channels than graph spaces. Pad zeros to keep shapes consistent.
    pad_cols = [f"__pad_space_{i}" for i in range(space_count - n_energy)]
    pad_df = pd.DataFrame(
        np.zeros((len(aligned), len(pad_cols)), dtype=np.float64),
        index=aligned.index,
        columns=pad_cols,
    )
    return pd.concat([aligned, pad_df], axis=1)


def _prepare_case_tensors(
    case_source,
    weather_scaler=None,
    energy_scaler=None,
    energy_alpha=80.0,
    normalize_weather=True,
    normalize_energy=True,
    use_multiscale_time_encoding=True,
    drop_original_time_feature=True,
    weather_encoding_strategy="base",
):
    case = load_case_data(case_source)
    if _is_preprocessed_pyg_case(case):
        base_graph = case["graph"]
        weather = case["weather"]
        energy = case["energy"]

        if not torch.is_tensor(weather):
            weather = torch.tensor(weather, dtype=torch.float)
        else:
            weather = weather.float()

        if not torch.is_tensor(energy):
            energy = torch.tensor(energy, dtype=torch.float)
        else:
            energy = energy.float()

        usable_steps = min(weather.shape[0], energy.shape[0])
        weather = weather[:usable_steps]
        energy = energy[:usable_steps]

        weather, scaler_weather = _normalize_weather_tensor(
            weather,
            None,
            normalize_weather=normalize_weather,
            weather_scaler=weather_scaler,
        )
        energy, scaler_energy = _normalize_energy_tensor(
            energy,
            normalize_energy=normalize_energy,
            energy_scaler=energy_scaler,
            energy_alpha=energy_alpha,
        )

        return {
            "graph": base_graph,
            "weather": weather,
            "energy": energy,
            "scaler_weather": scaler_weather,
            "scaler_energy": scaler_energy,
            "usable_steps": int(usable_steps),
        }

    weather_df = build_multiscale_time_features(
        case["weather"],
        use_multiscale_time_encoding=use_multiscale_time_encoding,
        drop_original_time_feature=drop_original_time_feature,
        weather_encoding_strategy=weather_encoding_strategy,
    )
    base_graph = build_hetero_graph(case["building"])
    space_count = int(base_graph["space"].x.size(0))
    energy_df = _align_energy_df_with_graph(case, case["energy"], space_count)

    weather = torch.tensor(weather_df.values, dtype=torch.float)
    energy = torch.tensor(energy_df.values, dtype=torch.float)

    usable_steps = min(weather.shape[0], energy.shape[0])
    weather = weather[:usable_steps]
    energy = energy[:usable_steps]

    weather, scaler_weather = _normalize_weather_tensor(
        weather,
        weather_df,
        normalize_weather=normalize_weather,
        weather_scaler=weather_scaler,
    )
    energy, scaler_energy = _normalize_energy_tensor(
        energy,
        normalize_energy=normalize_energy,
        energy_scaler=energy_scaler,
        energy_alpha=energy_alpha,
    )

    return {
        "graph": base_graph,
        "weather": weather,
        "energy": energy,
        "scaler_weather": scaler_weather,
        "scaler_energy": scaler_energy,
        "usable_steps": usable_steps,
    }


def build_preprocessed_pyg_case(
    case_source,
    case_key: Optional[str] = None,
    use_multiscale_time_encoding=True,
    drop_original_time_feature=True,
    weather_encoding_strategy="base",
):
    case = load_case_data(case_source)
    if _is_preprocessed_pyg_case(case):
        result = dict(case)
        if case_key is not None:
            result["case_key"] = str(case_key)
        return result

    weather_df = build_multiscale_time_features(
        case["weather"],
        use_multiscale_time_encoding=use_multiscale_time_encoding,
        drop_original_time_feature=drop_original_time_feature,
        weather_encoding_strategy=weather_encoding_strategy,
    )
    base_graph = build_hetero_graph(case["building"])
    space_count = int(base_graph["space"].x.size(0))
    energy_df = _align_energy_df_with_graph(case, case["energy"], space_count)

    weather = torch.tensor(weather_df.values, dtype=torch.float)
    energy = torch.tensor(energy_df.values, dtype=torch.float)
    usable_steps = min(int(weather.shape[0]), int(energy.shape[0]))

    preprocessed = {
        "__pyg_preprocessed__": True,
        "case_key": str(case_key) if case_key is not None else None,
        "graph": base_graph,
        "weather": weather[:usable_steps].contiguous(),
        "energy": energy[:usable_steps].contiguous(),
        "weather_columns": [str(c) for c in weather_df.columns],
        "weather_temp_indices": _infer_weather_temperature_indices(weather_df),
    }
    return preprocessed


def save_preprocessed_pyg_cases(
    data_dict,
    output_dir,
    overwrite=False,
    use_multiscale_time_encoding=True,
    drop_original_time_feature=True,
    weather_encoding_strategy="base",
    verbose=True,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for case_key in data_dict.keys():
        case_id = str(case_key)
        case_file = output_path / f"{case_id}.pt"
        if case_file.exists() and not overwrite:
            skipped += 1
            continue

        pyg_case = build_preprocessed_pyg_case(
            data_dict[case_key],
            case_key=case_id,
            use_multiscale_time_encoding=use_multiscale_time_encoding,
            drop_original_time_feature=drop_original_time_feature,
            weather_encoding_strategy=weather_encoding_strategy,
        )
        torch.save(pyg_case, case_file)
        saved += 1

    if verbose:
        print(
            f"Saved preprocessed PyG cases to {output_path} | saved={saved}, skipped={skipped}, total={len(data_dict)}"
        )

    return {
        "saved": saved,
        "skipped": skipped,
        "total": len(data_dict),
        "output_dir": str(output_path),
    }


def read_pyg_data(input_dir: str, n_files=None) -> dict:
    """Read preprocessed per-case PyG files from directory."""
    input_path = Path(input_dir)
    pt_files = sorted(input_path.glob("*.pt"))

    if isinstance(n_files, int):
        if n_files < 0 or n_files >= len(pt_files):
            raise IndexError(f"n_files index out of range: {n_files}, total files: {len(pt_files)}")
        pt_files = [pt_files[n_files]]
    elif isinstance(n_files, (list, tuple)) and len(n_files) == 2:
        a, b = n_files
        pt_files = pt_files[a:b]

    dataset = {}
    for f in pt_files:
        case_name = f.stem
        dataset[case_name] = load_case_data(f)

    print(f"Loaded {len(dataset)} PyG cases from {input_path}")
    return dataset


def _build_time_windows(tensor, day_length, day_stride):
    max_start = tensor.shape[0] - day_length
    if max_start < 0:
        return None
    return tensor.unfold(0, day_length, day_stride).transpose(1, 2).contiguous()


class GraphDataset(Dataset):
    """Unified graph dataset supporting hour/day sampling with eager loading."""

    def __init__(
        self,
        data_dict,
        split: str,
        split_ratio=(0.7, 0.15, 0.15),
        seed=42,
        shuffle=True,
        verbose=True,
        case_keys=None,
        weather_scaler=None,
        energy_scaler=None,
        energy_alpha=80.0,
        normalize_weather=True,
        normalize_energy=True,
        use_multiscale_time_encoding=True,
        drop_original_time_feature=True,
        weather_encoding_strategy="base",
        day_length=24,
        day_stride=24,
        granularity="hour",
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if granularity not in {"hour", "day"}:
            raise ValueError(f"Unsupported granularity: {granularity}")

        self.data_dict = data_dict
        self.split = split
        self.granularity = granularity
        self.day_length = int(day_length)
        self.day_stride = int(day_stride)
        self.weather_scaler = weather_scaler
        self.energy_scaler = energy_scaler
        self.energy_alpha = energy_alpha
        self.normalize_weather = normalize_weather
        self.normalize_energy = normalize_energy
        self.use_multiscale_time_encoding = use_multiscale_time_encoding
        self.drop_original_time_feature = drop_original_time_feature
        self.weather_encoding_strategy = weather_encoding_strategy

        if self.granularity == "day":
            if self.day_length <= 0:
                raise ValueError("day_length must be positive.")
            if self.day_stride <= 0:
                raise ValueError("day_stride must be positive.")

        self.cases = []
        self.idx_map = []

        rng = random.Random(seed)
        selected_cases, split_cases = _resolve_case_split(
            data_dict,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            case_keys=case_keys,
        )

        train_cases = split_cases["train"]
        val_cases = split_cases["val"]
        test_cases = split_cases["test"]
        n_cases = len(train_cases) + len(val_cases) + len(test_cases)

        for case_i, case_key in enumerate(selected_cases):
            case_idx = len(self.cases)
            prepared = self._prepare_case(case_key)
            self.cases.append(prepared)
            sample_count = self._append_eager_indices(case_idx, prepared)
            usable_steps = int(prepared["usable_steps"])

            if verbose:
                self._print_progress(split, case_i, len(selected_cases), usable_steps, sample_count)

        if shuffle:
            rng.shuffle(self.idx_map)

        if verbose:
            print(
                f"\nEager dataset initialized ({self.granularity}-level): "
                f"{len(self.idx_map)} samples from {len(selected_cases)} cases "
                f"(total {n_cases} cases: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test)."
            )

    def _prepare_case(self, case_key):
        prepared = _prepare_case_tensors(
            self.data_dict[case_key],
            weather_scaler=self.weather_scaler,
            energy_scaler=self.energy_scaler,
            energy_alpha=self.energy_alpha,
            normalize_weather=self.normalize_weather,
            normalize_energy=self.normalize_energy,
            use_multiscale_time_encoding=self.use_multiscale_time_encoding,
            drop_original_time_feature=self.drop_original_time_feature,
            weather_encoding_strategy=self.weather_encoding_strategy,
        )

        if self.granularity == "day":
            prepared["weather_windows"] = _build_time_windows(
                prepared["weather"], self.day_length, self.day_stride
            )
            prepared["energy_windows"] = _build_time_windows(
                prepared["energy"], self.day_length, self.day_stride
            )

        return prepared

    def _append_eager_indices(self, case_idx, prepared):
        if self.granularity == "hour":
            total = int(prepared["usable_steps"])
            for t in range(total):
                self.idx_map.append((case_idx, t))
            return total

        weather_windows = prepared.get("weather_windows")
        energy_windows = prepared.get("energy_windows")
        if weather_windows is None or energy_windows is None:
            return 0

        total = int(weather_windows.size(0))
        for day_idx in range(total):
            self.idx_map.append((case_idx, day_idx))
        return total

    def _print_progress(self, split, case_i, total_cases, usable_steps, sample_count):
        if self.granularity == "hour":
            print(
                f"\r'{split}'={len(self.idx_map):5d} | "
                f"Case {case_i+1:3d}/{total_cases}: Timesteps={usable_steps}",
                end="",
            )
        else:
            print(
                f"\r'{split}'={len(self.idx_map):5d} | "
                f"Case {case_i+1:3d}/{total_cases}: Steps={usable_steps}, DaySamples={sample_count}",
                end="",
            )

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        case_ref, time_idx = self.idx_map[idx]
        case = self.cases[case_ref]

        data = copy.copy(case["graph"])
        if self.granularity == "hour":
            data.weather = case["weather"][time_idx:time_idx + 1]
            data.energy = case["energy"][time_idx]
            return data

        weather_day = case["weather_windows"][time_idx]
        energy_day = case["energy_windows"][time_idx]
        data.weather = weather_day.unsqueeze(0)
        data.energy = energy_day.transpose(0, 1).contiguous()
        return data


class GraphHourlyDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        kwargs["granularity"] = "hour"
        super().__init__(*args, **kwargs)


class GraphDailyDataset(GraphDataset):
    def __init__(self, *args, **kwargs):
        kwargs["granularity"] = "day"
        super().__init__(*args, **kwargs)


def build_hetero_graph(building_dict):
    """
    Build PyG HeteroData from dict(nodes_df, edges_df, space_order)
    """
    data = HeteroData()

    face_feats = building_dict['face_feats']
    space_feats = building_dict['space_feats']

    ff_edges = building_dict['ff_edges']
    sf_edges = building_dict['sf_edges']
    sf_edge_attr = building_dict['sf_edge_attr']

    # Use MinMaxScaler to normalize node features
    scaler_face = MinMaxScaler()
    scaler_space = MinMaxScaler()
    face_feats = np.array(face_feats)

    face_geom = face_feats[:, :-1]      # s + n
    face_type = face_feats[:, -1:]       # 0 / 1, do not normalize

    face_geom = scaler_face.fit_transform(face_geom)

    face_feats = np.hstack([face_geom, face_type])
    data["face"].x = torch.tensor(face_feats, dtype=torch.float)
    space_feats = scaler_space.fit_transform(np.array(space_feats))
    data["space"].x = torch.tensor(space_feats, dtype=torch.float)

    # ---------- set edge_index ----------
    if ff_edges:
        data["face", "adj", "face"].edge_index = (torch.tensor(ff_edges, dtype=torch.long).t().contiguous())

    if sf_edges:
        data["face", "to", "space"].edge_index = (torch.tensor(sf_edges, dtype=torch.long).t().contiguous())
        data["face", "to", "space"].edge_attr = (torch.tensor(sf_edge_attr, dtype=torch.float))

    return data

def get_data_dir(input_dir: str):
    base = Path(input_dir)
    return {
        "weather": str(base / "weather"),
        "building": str(base / "graph"),
        "energy": str(base / "data")
    }

# Canonical weather names used by existing feature engineering logic.
_WEATHER_RENAME_MAP = {
    "dry_bulb": "db",
    "dry_bulb_temperature": "db",
    "relative_humidity": "rh",
    "global_horizontal_radiation": "ghr",
    "direct_normal_radiation": "dnr",
    "diffuse_horizontal_radiation": "dhr",
}


def _as_set(values: Optional[Union[Sequence, int, str]]) -> Optional[set]:
    if values is None:
        return None
    if isinstance(values, (str, int)):
        return {str(values)}
    return {str(v) for v in values}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_split(value: str) -> str:
    return str(value).strip().lower()


def _resolve_case_id_column(df: pd.DataFrame) -> str:
    for col in ("sample_id", "case_id"):
        if col in df.columns:
            return col
    raise ValueError("Split CSV must contain either 'sample_id' or 'case_id' column.")


def _apply_n_files_slice(df: pd.DataFrame, n_files=None) -> pd.DataFrame:
    if isinstance(n_files, int):
        if n_files < 0 or n_files >= len(df):
            raise IndexError(f"n_files index out of range: {n_files}, total files: {len(df)}")
        return df.iloc[[n_files]]

    if isinstance(n_files, (list, tuple)) and len(n_files) == 2:
        start, end = n_files
        return df.iloc[start:end]

    if n_files is None:
        return df

    raise TypeError("n_files must be None, int, or [start, end].")


def read_pack_manifest(
    pack_dir: Union[str, Path],
    case_ids: Optional[Union[Sequence[str], str]] = None,
    building_ids: Optional[Union[Sequence[Union[int, str]], Union[int, str]]] = None,
    weather_ids: Optional[Union[Sequence[str], str]] = None,
    n_files=None,
) -> pd.DataFrame:
    """Read and filter pack manifest by case/building/weather selectors."""
    pack_root = Path(pack_dir)
    manifest_path = pack_root / "manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "source_job_tag": str,
            "weather_id": str,
            "building_id": str,
            "energy_file": str,
        },
    )
    if "sample_id" not in manifest.columns:
        raise ValueError(f"manifest.csv missing required column: sample_id ({manifest_path})")

    required_cols = {"building_id", "weather_id", "energy_file"}
    missing = sorted(required_cols - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest.csv missing required columns: {missing}")

    for col in ["sample_id", "building_id", "weather_id", "energy_file"]:
        manifest[col] = manifest[col].astype(str).str.strip()

    case_set = _as_set(case_ids)
    building_set = _as_set(building_ids)
    weather_set = _as_set(weather_ids)

    sliced = _apply_n_files_slice(manifest.reset_index(drop=True), n_files=n_files)

    filtered = sliced.copy()
    if case_set is not None:
        filtered = filtered[filtered["sample_id"].astype(str).isin(case_set)]
    if building_set is not None:
        filtered = filtered[filtered["building_id"].astype(str).isin(building_set)]
    if weather_set is not None:
        filtered = filtered[filtered["weather_id"].astype(str).isin(weather_set)]

    return filtered.reset_index(drop=True)


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _to_weather_df(npz_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    values = np.asarray(npz_data["values"])
    columns = [str(c) for c in np.asarray(npz_data["columns"]).tolist()]

    df = pd.DataFrame(values, columns=columns)
    rename_map = {c: _WEATHER_RENAME_MAP[c.lower()] for c in df.columns if c.lower() in _WEATHER_RENAME_MAP}
    if rename_map:
        df = df.rename(columns=rename_map)

    if "time" not in {str(c).lower() for c in df.columns}:
        df.insert(0, "time", np.arange(1, len(df) + 1, dtype=np.int32))

    return df


def _to_energy_df(npz_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    values = np.asarray(npz_data["values"])
    columns = [str(c) for c in np.asarray(npz_data["columns"]).tolist()]
    return pd.DataFrame(values, columns=columns)


def _to_building_dict(npz_data: Dict[str, np.ndarray]) -> Dict[str, list]:
    required = ["ff_edges", "sf_edges", "sf_edge_attr", "face_feats", "space_feats"]
    missing = [k for k in required if k not in npz_data]
    if missing:
        raise ValueError(f"building npz missing required keys: {missing}")

    building = {
        "ff_edges": np.asarray(npz_data["ff_edges"]).tolist(),
        "sf_edges": np.asarray(npz_data["sf_edges"]).tolist(),
        "sf_edge_attr": np.asarray(npz_data["sf_edge_attr"]).tolist(),
        "face_feats": np.asarray(npz_data["face_feats"]).tolist(),
        "space_feats": np.asarray(npz_data["space_feats"]).tolist(),
    }

    if "valid_energy_spaces" in npz_data:
        building["valid_energy_spaces"] = [str(x) for x in np.asarray(npz_data["valid_energy_spaces"]).tolist()]

    return building


def _build_case_paths(pack_dir: Union[str, Path], row: pd.Series) -> Dict[str, Path]:
    root = Path(pack_dir)
    building_id = str(row["building_id"])
    weather_id = str(row["weather_id"])
    energy_file = str(row["energy_file"])

    return {
        "weather": root / "weather" / f"{weather_id}.npz",
        "building": root / "building" / f"{building_id}.npz",
        "energy": root / "energy" / energy_file,
    }


def read_data(
    input_dir: Union[str, Path],
    n_files=None,
    case_ids: Optional[Union[Sequence[str], str]] = None,
    building_ids: Optional[Union[Sequence[Union[int, str]], Union[int, str]]] = None,
    weather_ids: Optional[Union[Sequence[str], str]] = None,
    strict: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Read dataset from structured PACK directory."""
    filtered = read_pack_manifest(
        input_dir,
        case_ids=case_ids,
        building_ids=building_ids,
        weather_ids=weather_ids,
        n_files=n_files,
    )

    rows = [row for _, row in filtered.iterrows()]

    def _load_one_case(row: pd.Series):
        sample_id = str(row["sample_id"])
        paths = _build_case_paths(input_dir, row)

        missing = [name for name, p in paths.items() if not p.exists()]
        if missing:
            msg = f"Skip {sample_id}: missing files {missing}"
            if strict:
                raise FileNotFoundError(msg)
            return sample_id, None, msg

        weather_npz = _load_npz(paths["weather"])
        building_npz = _load_npz(paths["building"])
        energy_npz = _load_npz(paths["energy"])

        case_data = {
            "weather": _to_weather_df(weather_npz),
            "building": _to_building_dict(building_npz),
            "energy": _to_energy_df(energy_npz),
        }
        return sample_id, case_data, None

    result = {}
    loaded_items = (_load_one_case(row) for row in rows)

    for sample_id, case_data, warn_msg in loaded_items:
        if warn_msg:
            if verbose:
                print(warn_msg)
            continue
        result[sample_id] = case_data

    if verbose:
        print(f"Loaded {len(result)} cases from {Path(input_dir)}")

    return result


def _read_split_csv(split_csv_path: Union[str, Path]) -> Tuple[str, pd.DataFrame]:
    csv_path = Path(split_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    if "split" not in df.columns:
        raise ValueError(f"Split CSV missing required column: split ({csv_path})")

    case_col = _resolve_case_id_column(df)
    df = df[[case_col, "split"]].copy()
    df[case_col] = df[case_col].astype(str).str.strip()
    df["split"] = df["split"].astype(str).map(_normalize_split)
    df = df[df[case_col] != ""]
    df = df.drop_duplicates(subset=[case_col], keep="first").reset_index(drop=True)
    return case_col, df


def build_case_splits_from_csv(
    split_csv_path: Union[str, Path],
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Build case_splits directly from split CSV (train/val/test must exist)."""
    case_col, df = _read_split_csv(split_csv_path)

    train_ids = _dedupe_keep_order(df.loc[df["split"] == "train", case_col].tolist())
    val_ids = _dedupe_keep_order(df.loc[df["split"] == "val", case_col].tolist())
    test_ids = _dedupe_keep_order(df.loc[df["split"] == "test", case_col].tolist())

    if not train_ids:
        raise ValueError("No train cases found in split CSV.")
    if not val_ids:
        raise ValueError("No val cases found in split CSV.")
    if not test_ids:
        raise ValueError("No test cases found in split CSV.")

    train_set = set(train_ids)
    val_final = [c for c in val_ids if c not in train_set]
    val_set = set(val_final)
    test_final = [c for c in test_ids if c not in train_set and c not in val_set]

    if not val_final:
        raise ValueError("Validation split becomes empty after overlap filtering.")
    if not test_final:
        raise ValueError("Test split becomes empty after overlap filtering.")

    if verbose:
        print(
            "Split CSV resolved | "
            f"train={len(train_ids)}, val={len(val_final)}, test={len(test_final)}"
        )

    return {
        "train": train_ids,
        "val": val_final,
        "test": test_final,
    }


def read_split_data(
    pack_dir: Union[str, Path],
    split_csv_path: Union[str, Path],
    strict: bool = True,
    verbose: bool = True,
):
    """Load PACK cases referenced by split CSV and return data_dict + explicit case_splits."""
    case_splits = build_case_splits_from_csv(split_csv_path, verbose=verbose)

    all_case_ids = _dedupe_keep_order(
        case_splits["train"] + case_splits["val"] + case_splits["test"]
    )

    data_dict = read_data(
        pack_dir,
        case_ids=all_case_ids,
        strict=strict,
        verbose=verbose,
    )

    loaded = set(data_dict.keys())
    resolved_splits = {
        k: [case_id for case_id in v if case_id in loaded]
        for k, v in case_splits.items()
    }

    if verbose:
        missing = len(all_case_ids) - len(loaded)
        print(
            "Loaded split cases | "
            f"requested={len(all_case_ids)}, loaded={len(loaded)}, missing={missing}"
        )
        print(
            "Resolved split sizes | "
            f"train={len(resolved_splits['train'])}, "
            f"val={len(resolved_splits['val'])}, "
            f"test={len(resolved_splits['test'])}"
        )

    return data_dict, resolved_splits
