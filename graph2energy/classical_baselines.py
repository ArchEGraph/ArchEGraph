from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASELINE_NAMES = ("DegreeHourRidge", "GeometryHistGBR", "RCInspiredRidge")
WEATHER_ALIASES = {
    "dry_bulb": "dry_bulb",
    "dry_bulb_temperature": "dry_bulb",
    "db": "dry_bulb",
    "dew_point": "dew_point",
    "relative_humidity": "relative_humidity",
    "rh": "relative_humidity",
    "global_horizontal_radiation": "global_horizontal_radiation",
    "ghr": "global_horizontal_radiation",
    "direct_normal_radiation": "direct_normal_radiation",
    "dnr": "direct_normal_radiation",
    "diffuse_horizontal_radiation": "diffuse_horizontal_radiation",
    "dhr": "diffuse_horizontal_radiation",
    "wind_speed": "wind_speed",
}
REQUIRED_WEATHER = (
    "dry_bulb",
    "dew_point",
    "relative_humidity",
    "global_horizontal_radiation",
    "direct_normal_radiation",
    "diffuse_horizontal_radiation",
    "wind_speed",
)
ZONE_FEATURE_NAMES = (
    "centroid_x_normalized",
    "centroid_y_normalized",
    "centroid_z_normalized",
    "log_incident_face_count",
    "external_face_fraction",
    "log_incident_area_proxy",
    "log_external_area_proxy",
    "log_mean_face_area_proxy",
    "area_weighted_abs_normal_x",
    "area_weighted_abs_normal_y",
    "area_weighted_abs_normal_z",
    "mean_face_level",
    "log_building_space_count",
    "log_building_face_count",
)


@dataclass(frozen=True)
class WeatherData:
    values: np.ndarray
    columns: tuple[str, ...]


def normalize_weather(values, columns) -> WeatherData:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Weather values must be 2D, got shape {array.shape}.")

    normalized_columns = tuple(
        WEATHER_ALIASES.get(str(column).strip().lower(), str(column).strip().lower())
        for column in columns
    )
    missing = [name for name in REQUIRED_WEATHER if name not in normalized_columns]
    if missing:
        raise ValueError(f"Weather data is missing required columns: {missing}")

    indices = [normalized_columns.index(name) for name in REQUIRED_WEATHER]
    selected = array[:, indices]
    if not np.isfinite(selected).all():
        raise ValueError("Weather data contains non-finite values.")
    return WeatherData(values=selected, columns=REQUIRED_WEATHER)


def aggregate_zone_geometry(building: dict) -> np.ndarray:
    face_features = np.asarray(building["face_feats"], dtype=np.float64)
    space_features = np.asarray(building["space_feats"], dtype=np.float64)
    edges = np.asarray(building["sf_edges"], dtype=np.int64)
    edge_attributes = np.asarray(building["sf_edge_attr"], dtype=np.float64).reshape(-1)

    if face_features.ndim != 2 or face_features.shape[1] < 7:
        raise ValueError(f"Expected face_feats [N, >=7], got {face_features.shape}.")
    if space_features.ndim != 2 or space_features.shape[1] < 3:
        raise ValueError(f"Expected space_feats [N, >=3], got {space_features.shape}.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected sf_edges [E, 2], got {edges.shape}.")
    if len(edge_attributes) != len(edges):
        raise ValueError("sf_edge_attr rows must match sf_edges rows.")

    n_faces = len(face_features)
    n_spaces = len(space_features)
    if n_faces == 0 or n_spaces == 0:
        raise ValueError("Building must contain at least one face and one space.")
    if (
        (edges[:, 0] < 0).any()
        or (edges[:, 0] >= n_faces).any()
        or (edges[:, 1] < 0).any()
        or (edges[:, 1] >= n_spaces).any()
    ):
        raise ValueError("sf_edges contains out-of-range face or space indices.")

    centroids = space_features[:, :3]
    centroid_min = centroids.min(axis=0)
    centroid_span = np.maximum(centroids.max(axis=0) - centroid_min, 1e-9)
    normalized_centroids = (centroids - centroid_min) / centroid_span

    dimensions = np.abs(face_features[:, :3])
    two_largest = np.sort(dimensions, axis=1)[:, -2:]
    face_area_proxy = np.prod(two_largest, axis=1)
    face_normals = np.abs(face_features[:, 3:6])
    face_level = face_features[:, 6]

    result = np.zeros((n_spaces, len(ZONE_FEATURE_NAMES)), dtype=np.float64)
    result[:, :3] = normalized_centroids
    result[:, 12] = np.log1p(n_spaces)
    result[:, 13] = np.log1p(n_faces)

    for space_index in range(n_spaces):
        edge_rows = np.flatnonzero(edges[:, 1] == space_index)
        if edge_rows.size == 0:
            continue
        face_indices = edges[edge_rows, 0]
        areas = face_area_proxy[face_indices]
        total_area = max(float(areas.sum()), 1e-12)
        external = edge_attributes[edge_rows] > 0.5
        external_area = float(areas[external].sum())

        result[space_index, 3] = np.log1p(len(edge_rows))
        result[space_index, 4] = float(external.mean())
        result[space_index, 5] = np.log1p(total_area)
        result[space_index, 6] = np.log1p(external_area)
        result[space_index, 7] = np.log1p(float(areas.mean()))
        result[space_index, 8:11] = np.average(
            face_normals[face_indices],
            axis=0,
            weights=np.maximum(areas, 1e-12),
        )
        result[space_index, 11] = float(face_level[face_indices].mean())

    if not np.isfinite(result).all():
        raise ValueError("Aggregated zone geometry contains non-finite values.")
    return result


def _cyclical_time_features(time_indices: np.ndarray) -> np.ndarray:
    time = np.asarray(time_indices, dtype=np.float64)
    hour_angle = 2.0 * np.pi * time / 24.0
    year_angle = 2.0 * np.pi * time / 8760.0
    return np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(year_angle),
            np.cos(year_angle),
        ]
    )


def _exponential_filter(values: np.ndarray, time_constant: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Exponential filter expects a non-empty 1D sequence.")
    alpha = 1.0 - np.exp(-1.0 / float(time_constant))
    filtered = np.empty_like(array)
    filtered[0] = array[0]
    for index in range(1, len(array)):
        filtered[index] = filtered[index - 1] + alpha * (
            array[index] - filtered[index - 1]
        )
    return filtered


def prepare_dynamic_features(
    weather: WeatherData,
    degree_base_temperature: float = 18.0,
) -> dict[str, np.ndarray]:
    values = weather.values
    dry_bulb = values[:, 0]
    solar_total = values[:, 3] + values[:, 4] + values[:, 5]
    heating_degree_hours = np.maximum(float(degree_base_temperature) - dry_bulb, 0.0)
    cooling_degree_hours = np.maximum(dry_bulb - float(degree_base_temperature), 0.0)

    common = np.column_stack(
        [
            values,
            heating_degree_hours,
            cooling_degree_hours,
            _cyclical_time_features(np.arange(len(values))),
        ]
    )
    degree = np.column_stack(
        [
            heating_degree_hours,
            cooling_degree_hours,
            values[:, 2:7],
            _cyclical_time_features(np.arange(len(values))),
        ]
    )

    rc_columns = []
    for time_constant in (6.0, 24.0, 72.0):
        filtered_temperature = _exponential_filter(dry_bulb, time_constant)
        filtered_solar = _exponential_filter(solar_total, time_constant)
        rc_columns.extend(
            [
                filtered_temperature,
                filtered_solar,
                np.maximum(float(degree_base_temperature) - filtered_temperature, 0.0),
                np.maximum(filtered_temperature - float(degree_base_temperature), 0.0),
            ]
        )
    rc = np.column_stack([common, *rc_columns])
    return {
        "DegreeHourRidge": degree,
        "GeometryHistGBR": common,
        "RCInspiredRidge": rc,
    }


def build_feature_rows(
    baseline_name: str,
    dynamic_features: dict[str, np.ndarray],
    zone_features: np.ndarray,
    time_indices: np.ndarray,
    zone_indices: np.ndarray,
) -> np.ndarray:
    if baseline_name not in BASELINE_NAMES:
        raise ValueError(
            f"Unknown baseline {baseline_name}. Expected one of {BASELINE_NAMES}."
        )

    time_indices = np.asarray(time_indices, dtype=np.int64)
    zone_indices = np.asarray(zone_indices, dtype=np.int64)
    if time_indices.shape != zone_indices.shape:
        raise ValueError("time_indices and zone_indices must have identical shapes.")

    dynamic = dynamic_features[baseline_name][time_indices]
    static = zone_features[zone_indices]
    geometry_drivers = static[:, [4, 5, 6]]

    if baseline_name == "DegreeHourRidge":
        forcing = dynamic[:, [0, 1, 2, 3, 4]]
        interactions = np.einsum("ni,nj->nij", forcing, geometry_drivers).reshape(
            len(dynamic), -1
        )
        features = np.column_stack([dynamic, static, interactions])
    elif baseline_name == "RCInspiredRidge":
        thermal_response = dynamic[:, -12:]
        interactions = np.einsum(
            "ni,nj->nij", thermal_response, geometry_drivers
        ).reshape(len(dynamic), -1)
        features = np.column_stack([dynamic, static, interactions])
    else:
        features = np.column_stack([dynamic, static])

    if not np.isfinite(features).all():
        raise ValueError("Classical baseline features contain non-finite values.")
    return features.astype(np.float32, copy=False)


def build_estimator(baseline_name: str, seed: int):
    if baseline_name in {"DegreeHourRidge", "RCInspiredRidge"}:
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    if baseline_name == "GeometryHistGBR":
        return HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=int(seed),
        )
    raise ValueError(f"Unknown baseline {baseline_name}.")


def sample_case_rows(
    n_time_steps: int,
    n_zones: int,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = int(n_time_steps) * int(n_zones)
    if total_rows <= 0:
        raise ValueError("Cases must contain at least one time step and one zone.")

    requested = min(max(int(sample_count), 1), total_rows)
    if requested == total_rows:
        flat_indices = np.arange(total_rows, dtype=np.int64)
        return np.divmod(flat_indices, int(n_zones))
    flat_indices = rng.choice(total_rows, size=requested, replace=False)
    time_indices, zone_indices = np.divmod(flat_indices, int(n_zones))
    return time_indices, zone_indices


class RegressionMetrics:
    def __init__(self):
        self.count = 0
        self.sum_target = 0.0
        self.sum_target_squared = 0.0
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0

    def update(self, target, prediction) -> None:
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)
        prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if target_array.shape != prediction_array.shape:
            raise ValueError("Target and prediction shapes must match.")
        if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
            raise ValueError("Metric inputs contain non-finite values.")

        residual = target_array - prediction_array
        self.count += int(target_array.size)
        self.sum_target += float(target_array.sum())
        self.sum_target_squared += float(np.square(target_array).sum())
        self.sum_squared_error += float(np.square(residual).sum())
        self.sum_absolute_error += float(np.abs(residual).sum())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("Cannot compute metrics without observations.")
        mse = self.sum_squared_error / self.count
        target_ss = self.sum_target_squared - self.sum_target**2 / self.count
        r2 = 1.0 - self.sum_squared_error / target_ss if target_ss > 0.0 else 0.0
        return {
            "count": self.count,
            "mse": float(mse),
            "mae": float(self.sum_absolute_error / self.count),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2),
        }
