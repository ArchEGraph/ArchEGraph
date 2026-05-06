import os
import json
import torch
from datetime import datetime
from pathlib import Path

ARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ARCH_ROOT / "cache"


def _normalize_run_name(run_name):
    if run_name is None:
        return None
    text = str(run_name).strip()
    if not text:
        return None

    # Keep path-safe characters for cross-platform folders.
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")

    normalized = "".join(safe).strip("._")
    return normalized or None


def _artifact_stem(run_name, timestamp):
    normalized = _normalize_run_name(run_name)
    if normalized:
        return normalized
    return str(timestamp)


def _resolve_cache_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ARCH_ROOT / path
    return path


def _resolve_run_dir(base_dir, run_name=None, timestamp=None):
    cache_root = _resolve_cache_path(base_dir or DEFAULT_CACHE_ROOT)
    artifact_stem = _artifact_stem(run_name, timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    return cache_root, cache_root / artifact_stem, artifact_stem


def _make_json_serializable(value):
    """Convert config values to JSON-serializable objects."""
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


class ModelSaver:
    """
    Save best model during training
    """
    def __init__(
        self,
        save_dir="cache",
        config=None,
        timestamp=None,
        enable_file=True,
        run_name=None,
    ):
        """
        Args:
            save_dir: Directory to save models
            config: Configuration dict for filename generation, e.g. {'lr': 5e-3, 'batch_size': 1536}
        """
        self.run_name = _normalize_run_name(run_name)
        self.config = config or {}
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_file = bool(enable_file)
        self.cache_root, self.run_dir, self.artifact_stem = _resolve_run_dir(
            save_dir,
            run_name=self.run_name,
            timestamp=self.timestamp,
        )
        self.best_state_dict = None

        if self.enable_file:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.run_dir / "model.pth" if self.enable_file else None
        
        self.best_metric = float('inf')
        self.best_epoch = 0
        
    def save(self, model, metric, epoch, mode='min'):
        """
        Save best model based on metric
        
        Args:
            model: PyTorch model
            metric: Current metric value (e.g., RMSE, loss)
            epoch: Current epoch
            mode: 'min' for lower is better, 'max' for higher is better
        
        Returns:
            bool: Whether the model was saved
        """
        is_better = False
        
        if mode == 'min':
            is_better = metric < self.best_metric
        elif mode == 'max':
            is_better = metric > self.best_metric
        else:
            raise ValueError("mode must be 'min' or 'max'")
        
        if is_better:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if self.enable_file and self.model_path is not None:
                torch.save(model.state_dict(), self.model_path)
            return True
        
        return False
    
    def load(self, model):
        """
        Load best model
        
        Args:
            model: PyTorch model
        
        Returns:
            model: Model with loaded weights
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print("Model loaded from in-memory best checkpoint")
            print(f"Best metric: {self.best_metric:.6f} at epoch {self.best_epoch}")
        elif self.enable_file and self.model_path is not None and self.model_path.exists():
            model.load_state_dict(torch.load(self.model_path))
            print(f"Model loaded from {self.model_path}")
            print(f"Best metric: {self.best_metric:.6f} at epoch {self.best_epoch}")
        else:
            print("Warning: No saved best model found. Using current model weights.")
        
        return model
    
    def get_path(self):
        """Return model save path"""
        return str(self.model_path) if self.model_path is not None else "(cache disabled)"


class TrainingLogger:
    """
    Record all metrics during training and save as JSON
    """
    def __init__(
        self,
        log_dir="cache",
        config=None,
        timestamp=None,
        enable_file=True,
        run_name=None,
    ):
        """
        Args:
            log_dir: Directory to save logs
            config: Configuration dict for filename generation
        """
        self.run_name = _normalize_run_name(run_name)
        self.config = config or {}
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_file = bool(enable_file)
        self.cache_root, self.run_dir, self.artifact_stem = _resolve_run_dir(
            log_dir,
            run_name=self.run_name,
            timestamp=self.timestamp,
        )

        if self.enable_file:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "metrics.json" if self.enable_file else None
        
        self.metrics_history = []
        self.test_metrics = None

    def _flush(self):
        if not self.enable_file or self.log_path is None:
            return

        payload = {
            "history": _make_json_serializable(self.metrics_history),
        }
        if self.test_metrics is not None:
            payload["test"] = _make_json_serializable(self.test_metrics)

        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
    def log(self, epoch, metrics_dict):
        """
        Log metrics for one epoch
        
        Args:
            epoch: Current epoch number
            metrics_dict: Metrics dict, e.g. {'train_loss': 0.1, 'val_loss': 0.2, 'val_r2': 0.9}
        """
        record = {"epoch": epoch}
        record.update(dict(metrics_dict or {}))
        self.metrics_history.append(record)
        self._flush()
    
    def log_test_results(self, test_metrics):
        """
        Log test results to a separate file
        
        Args:
            test_metrics: Test metrics dict, e.g. {'test_r2': 0.95, 'test_mae': 0.1, 'test_rmse': 0.15}
        """
        if not self.enable_file or self.log_path is None:
            print("Test results file save skipped (cache disabled).")
            return

        self.test_metrics = dict(test_metrics or {})
        self._flush()
        print(f"Test results saved to {self.log_path}")
    
    def get_best_epoch(self, metric_name, mode='min'):
        """
        Get best epoch for specified metric
        
        Args:
            metric_name: Metric name, e.g. 'val_rmse'
            mode: 'min' or 'max'
        
        Returns:
            tuple: (best_epoch, best_value)
        """
        if not self.metrics_history:
            return None, None
        
        values = [m.get(metric_name, float('inf') if mode == 'min' else float('-inf')) 
                  for m in self.metrics_history]
        
        if mode == 'min':
            best_idx = min(range(len(values)), key=lambda i: values[i])
        else:
            best_idx = max(range(len(values)), key=lambda i: values[i])
        
        best_epoch = self.metrics_history[best_idx]['epoch']
        best_value = values[best_idx]
        
        return best_epoch, best_value

    def _infer_metric_mode(self, metric_name):
        """Infer whether a metric should be minimized, maximized, or skipped."""
        name = metric_name.lower()

        # Metadata / control-like fields that should not be ranked as best.
        skip_tokens = [
            'threshold',
            'quantile',
            'count',
            'sum_true',
            'sum_pred',
            'learning_rate',
            'lr',
        ]
        if any(token in name for token in skip_tokens):
            return None

        # Metrics where larger is better.
        maximize_tokens = [
            'r2',
            'spearman',
            'pearson',
            'accuracy',
            'auc',
            'f1',
            'precision',
            'recall',
        ]
        if any(token in name for token in maximize_tokens):
            return 'max'

        # Error-like metrics where smaller is better.
        minimize_tokens = [
            'loss',
            'mae',
            'rmse',
            'mse',
            'mape',
            'error',
            'abs_error',
            'max_error',
        ]
        if any(token in name for token in minimize_tokens):
            return 'min'

        # Safe default: minimize unknown regression metrics.
        return 'min'
    
    def summary(self):
        """
        Print training summary
        """
        if not self.metrics_history:
            print("No training history recorded.")
            return
        
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        if self.enable_file and self.log_path is not None:
            print(f"Log saved to: {self.log_path}")
        else:
            print("Log file save: disabled")
        print(f"Total epochs: {len(self.metrics_history)}")
        
        # Print best values for each metric
        metric_keys = [k for k in self.metrics_history[0].keys() if k != 'epoch']
        for metric in metric_keys:
            mode = self._infer_metric_mode(metric)
            if mode is None:
                continue
            
            best_epoch, best_value = self.get_best_epoch(metric, mode)
            if best_epoch is not None:
                print(f"Best {metric}: {best_value:.6f} at epoch {best_epoch}")
        
        print(f"{'='*60}\n")
    
    def get_path(self):
        """Return log save path"""
        return str(self.log_path) if self.log_path is not None else "(cache disabled)"


class ConfigSaver:
    """Save training config as JSON with aligned timestamp."""

    def __init__(self, config_dir="cache", timestamp=None, enable_file=True, run_name=None):
        self.run_name = _normalize_run_name(run_name)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_file = bool(enable_file)
        self.cache_root, self.run_dir, self.artifact_stem = _resolve_run_dir(
            config_dir,
            run_name=self.run_name,
            timestamp=self.timestamp,
        )

        if self.enable_file:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.config_path = self.run_dir / "config.json"
        else:
            self.config_path = None

    def save(self, config):
        if not self.enable_file or self.config_path is None:
            return
        serializable_config = _make_json_serializable(config or {})
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, ensure_ascii=False, indent=2)

    def get_path(self):
        return str(self.config_path) if self.config_path is not None else "(cache disabled)"


def create_logger_and_saver(config=None, save_dir="cache", log_dir="cache"):
    """
    Convenience function: create both ModelSaver and TrainingLogger
    
    Args:
        config: Configuration dict
        save_dir: Directory to save models
        log_dir: Directory to save logs
    
    Returns:
        tuple: (model_saver, training_logger)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _normalize_run_name((config or {}).get("run_name"))
    model_saver = ModelSaver(
        save_dir=save_dir,
        config=config,
        timestamp=timestamp,
        run_name=run_name,
    )
    training_logger = TrainingLogger(
        log_dir=log_dir,
        config=config,
        timestamp=timestamp,
        run_name=run_name,
    )
    
    
    return model_saver, training_logger


def create_training_artifacts(
    config=None,
    model_dir="cache",
    log_dir="cache",
    config_dir="cache",
    enable_cache=True,
):
    """Create model saver, logger and config saver with the same timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _normalize_run_name((config or {}).get("run_name"))
    model_saver = ModelSaver(
        save_dir=model_dir,
        config=config,
        timestamp=timestamp,
        enable_file=enable_cache,
        run_name=run_name,
    )
    training_logger = TrainingLogger(
        log_dir=log_dir,
        config=config,
        timestamp=timestamp,
        enable_file=enable_cache,
        run_name=run_name,
    )
    config_saver = ConfigSaver(
        config_dir=config_dir,
        timestamp=timestamp,
        enable_file=enable_cache,
        run_name=run_name,
    )

    if enable_cache:
        print(f"Model will be saved to: {model_saver.get_path()}")
        print(f"Training log will be saved to: {training_logger.get_path()}")
        print(f"Config will be saved to: {config_saver.get_path()}")
        if run_name:
            print(f"Run name: {run_name}")
    else:
        print("Cache artifact saving is disabled (model/log/config).")

    return model_saver, training_logger, config_saver
