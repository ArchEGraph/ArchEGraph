from __future__ import annotations

import torch
import torch.nn.functional as F


def build_all_pairs(n: int, device: torch.device) -> torch.Tensor:
    rows, cols = torch.triu_indices(n, n, offset=1, device=device)
    return torch.stack([rows, cols], dim=0)


def sample_ff_pairs(gt_adj: torch.Tensor, neg_per_pos: float = 1.0):
    n = gt_adj.size(0)
    device = gt_adj.device
    upper = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), diagonal=1)
    pos = ((gt_adj > 0.5) & upper).nonzero(as_tuple=False)
    neg = ((~(gt_adj > 0.5)) & upper).nonzero(as_tuple=False)

    if pos.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    n_pos = pos.size(0)
    n_neg = int(max(1, round(n_pos * neg_per_pos)))
    n_neg = min(n_neg, neg.size(0))
    pick = neg[torch.randperm(neg.size(0), device=device)[:n_neg]]

    pairs = torch.cat([pos, pick], dim=0)
    labels = torch.cat(
        [
            torch.ones((n_pos,), device=device, dtype=torch.float32),
            torch.zeros((n_neg,), device=device, dtype=torch.float32),
        ],
        dim=0,
    )
    return pairs.t().contiguous(), labels


def sample_sf_pairs(gt_sf_adj: torch.Tensor, gt_space_exists: torch.Tensor, neg_per_pos: float = 2.0):
    device = gt_sf_adj.device
    active = (gt_space_exists > 0.5).nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    sub = gt_sf_adj[active] > 0.5
    pos = sub.nonzero(as_tuple=False)
    neg = (~sub).nonzero(as_tuple=False)
    if pos.size(0) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    n_pos = pos.size(0)
    n_neg = int(max(1, round(n_pos * neg_per_pos)))
    n_neg = min(n_neg, neg.size(0))
    pick = neg[torch.randperm(neg.size(0), device=device)[:n_neg]]

    all_pairs = torch.cat([pos, pick], dim=0)
    labels = torch.cat(
        [
            torch.ones((n_pos,), dtype=torch.float32, device=device),
            torch.zeros((n_neg,), dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    s_idx = active[all_pairs[:, 0]]
    f_idx = all_pairs[:, 1]
    return torch.stack([s_idx, f_idx], dim=0), labels


def _auto_pos_weight(labels: torch.Tensor, mask: torch.Tensor | None = None, max_w: float = 100.0):
    labels = labels.float()
    if mask is None:
        mask = torch.ones_like(labels)
    else:
        mask = mask.float()

    pos = (labels * mask).sum()
    neg = ((1.0 - labels) * mask).sum()
    if float(pos.item()) <= 0.0:
        return 1.0
    w = float((neg / (pos + 1e-12)).item())
    return max(1.0, min(max_w, w))


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None):
    used_pos = _auto_pos_weight(labels, mask)
    pw = torch.tensor(used_pos, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pw, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.float()
    denom = mask.sum()
    if float(denom.item()) <= 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return (loss * mask).sum() / (denom + 1e-12)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5, mask: torch.Tensor | None = None):
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()
    labels = labels.float()
    if mask is None:
        mask = torch.ones_like(labels)
    else:
        mask = mask.float()

    tp = (pred * labels * mask).sum().item()
    fp = (pred * (1.0 - labels) * mask).sum().item()
    fn = ((1.0 - pred) * labels * mask).sum().item()
    correct = ((pred == labels).float() * mask).sum().item()
    total = mask.sum().item()
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float((2 * tp) / (2 * tp + fp + fn + 1e-12))
    acc = float(correct / (total + 1e-12))
    return precision, recall, f1, acc


def run_epoch(model, loader, optimizer, device, train: bool, metric_thr: float = 0.5):
    ff_w = 1.0
    s_w = 1.0
    sf_w = 2.0

    if train:
        model.train()
    else:
        model.eval()

    total = {
        "loss": 0.0,
        "ff_f1": 0.0,
        "ff_acc": 0.0,
        "ff_recall": 0.0,
        "s_f1": 0.0,
        "s_acc": 0.0,
        "s_recall": 0.0,
        "sf_f1": 0.0,
        "sf_acc": 0.0,
        "sf_recall": 0.0,
    }
    n_samples = 0

    for batch in loader:
        samples = batch if isinstance(batch, list) else [batch]
        for sample in samples:
            x_face = sample.x_face.to(device, non_blocking=True)
            gt_ff_adj = sample.gt_ff_adj.to(device, non_blocking=True)
            gt_space_exists = sample.gt_space_exists.to(device, non_blocking=True)
            gt_sf_adj = sample.gt_sf_adj.to(device, non_blocking=True)

            with torch.set_grad_enabled(train):
                h_face, s_logits, sf_logits = model.predict_all(x_face)

                ff_idx, ff_labels = sample_ff_pairs(gt_ff_adj, neg_per_pos=1.0)
                if ff_idx.numel() == 0:
                    ff_loss = torch.tensor(0.0, device=device)
                    ff_f1 = ff_acc = ff_recall = 0.0
                else:
                    ff_logit = model.ff_logits(h_face, ff_idx)
                    ff_loss = F.binary_cross_entropy_with_logits(ff_logit, ff_labels)
                    _, ff_recall, ff_f1, ff_acc = binary_metrics(ff_logit.detach(), ff_labels, thr=metric_thr)

                s_loss = masked_bce(s_logits, gt_space_exists)
                _, s_recall, s_f1, s_acc = binary_metrics(s_logits.detach(), gt_space_exists, thr=metric_thr)

                sf_idx, sf_labels = sample_sf_pairs(gt_sf_adj, gt_space_exists, neg_per_pos=2.0)
                if sf_idx.numel() == 0:
                    sf_loss = torch.tensor(0.0, device=device)
                    sf_f1 = sf_acc = sf_recall = 0.0
                else:
                    sf_pair_logit = sf_logits[sf_idx[0], sf_idx[1]]
                    sf_loss = F.binary_cross_entropy_with_logits(sf_pair_logit, sf_labels)
                    sf_mask = gt_space_exists.unsqueeze(1).expand_as(gt_sf_adj)
                    _, sf_recall, sf_f1, sf_acc = binary_metrics(sf_logits.detach(), gt_sf_adj, thr=metric_thr, mask=sf_mask)

                loss = ff_w * ff_loss + s_w * s_loss + sf_w * sf_loss

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total["loss"] += float(loss.item())
            total["ff_f1"] += float(ff_f1)
            total["ff_acc"] += float(ff_acc)
            total["ff_recall"] += float(ff_recall)
            total["s_f1"] += float(s_f1)
            total["s_acc"] += float(s_acc)
            total["s_recall"] += float(s_recall)
            total["sf_f1"] += float(sf_f1)
            total["sf_acc"] += float(sf_acc)
            total["sf_recall"] += float(sf_recall)
            n_samples += 1

    if n_samples == 0:
        return {k: 0.0 for k in total}
    return {k: v / n_samples for k, v in total.items()}
