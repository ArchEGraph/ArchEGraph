import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def batch_mse_loss(pred: torch.Tensor, target: torch.Tensor):
    loss = F.mse_loss(pred, target)
    n = int(target.numel())
    return loss, float(loss.item()) * max(n, 1), max(n, 1)


def flatten_numpy(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.detach().cpu().reshape(-1).numpy()
    target_np = target.detach().cpu().reshape(-1).numpy()
    return pred_np, target_np


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
