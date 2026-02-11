import numpy as np
import torch


def lsd_loss(pred_db: torch.Tensor, target_db: torch.Tensor) -> torch.Tensor:
    """
    pred_db, target_db: (B, 793, 2, F)
    LSD: mean over batch, directions, ears
    """
    diff = pred_db - target_db
    lsd_per_dir_ear = torch.sqrt(torch.mean(diff ** 2, dim=-1))  # (B,793,2)
    return torch.mean(lsd_per_dir_ear)


def lsd_loss_unseen(pred_db: torch.Tensor, target_db: torch.Tensor, unseen_didxs) -> torch.Tensor:
    """
    Evaluate LSD only on unseen directions.
    """
    idx = torch.as_tensor(unseen_didxs, dtype=torch.long, device=pred_db.device)
    pred_u = pred_db.index_select(dim=1, index=idx)     # (B, Nu, 2, F)
    tgt_u  = target_db.index_select(dim=1, index=idx)   # (B, Nu, 2, F)
    diff = pred_u - tgt_u
    lsd_per_dir_ear = torch.sqrt(torch.mean(diff ** 2, dim=-1))  # (B,Nu,2)
    return torch.mean(lsd_per_dir_ear)


def hrtf_shape_loss(pred_db: torch.Tensor, gt_db: torch.Tensor, lambda_grad: float = 1.0, grad_type: str = "l1"):
    """
    pred_db, gt_db: (B,793,2,F) in dB domain
    loss = LSD + lambda * gradient_loss_along_frequency
    """
    diff = pred_db - gt_db
    lsd_per = torch.sqrt(torch.mean(diff**2, dim=-1))  # (B,793,2)
    loss_val = torch.mean(lsd_per)

    grad_pred = torch.diff(pred_db, dim=-1)            # (B,793,2,F-1)
    grad_gt   = torch.diff(gt_db, dim=-1)

    if grad_type == "l2":
        loss_grad = torch.mean((grad_pred - grad_gt) ** 2)
    else:
        loss_grad = torch.mean(torch.abs(grad_pred - grad_gt))

    return loss_val + lambda_grad * loss_grad

# =========================================================
# Broadband ILD
# =========================================================
def ild_broadband_from_mag_db(spec_db, eps=1e-12):
    """
    spec_db: (D, 2, F) in dB
    return: (D,) broadband ILD per direction
    """
    mag = 10.0 ** (spec_db / 20.0)
    power = mag ** 2
    E = np.sum(power, axis=-1)
    level_db = 10.0 * np.log10(E + eps)
    return level_db[:, 0] - level_db[:, 1]


def ild_mae(pred_db, gt_db):
    """
    pred_db, gt_db: (B, 793, 2, F)
    """
    B = pred_db.shape[0]
    total = 0.0

    for b in range(B):
        pred_ild = ild_broadband_from_mag_db(pred_db[b])
        gt_ild = ild_broadband_from_mag_db(gt_db[b])
        total += np.mean(np.abs(pred_ild - gt_ild))

    return total / B