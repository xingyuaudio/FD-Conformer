# src/eval.py

import yaml
import numpy as np
import torch

from datasets.sparse_sets import get_sparse_set
from datasets.sonicom_dataset import SonicomHLogDataset
from models.hrtf_upsampler import HRTFMagUpsampler
from utils.metrics import (
    lsd_loss_unseen,
    ild_mae,
)
from utils.seed import seed_everything


# =========================================================
# Load config
# =========================================================
def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# =========================================================
# Build model
# =========================================================
def build_model(cfg, n_sparse):

    mcfg = cfg["model"]

    model = HRTFMagUpsampler(
        n_sparse=n_sparse,
        n_dense=cfg["data"]["n_dirs"],
        n_freq=cfg["data"]["n_freq"],
        variant=mcfg["variant"],
        d_model=mcfg["d_model"],
        d_ff=mcfg["d_ff"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        conv_kernel=mcfg["conv_kernel"],
        head_hidden=mcfg["head_hidden"],
        dropout=mcfg["dropout"],
        use_freq_pe=mcfg["use_freq_pe"],
        spatial_act=mcfg["spatial_act"],
    )

    return model


# =========================================================
# Evaluation loop
# =========================================================
def evaluate(model, loader, unseen_didxs, device):

    model.eval()
    total_lsd = 0.0
    total_ild = 0.0

    with torch.no_grad():
        for x_sparse, y_dense in loader:

            x_sparse = x_sparse.to(device)
            y_dense = y_dense.to(device)

            pred = model(x_sparse)

            # LSD (unseen directions only)
            total_lsd += lsd_loss_unseen(
                pred, y_dense, unseen_didxs
            ).item()

            # ILD (full directions)
            pred_np = pred.cpu().numpy()
            gt_np = y_dense.cpu().numpy()
            total_ild += ild_mae(pred_np, gt_np)

    return total_lsd / len(loader), total_ild / len(loader)


# =========================================================
# Main
# =========================================================
def main(cfg_path: str, ckpt_path: str):

    cfg = load_cfg(cfg_path)
    seed_everything(cfg["seed"])

    device = torch.device(
        "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print("Using device:", device)

    # -------------------------
    # Load data
    # -------------------------
    data = np.load(cfg["data"]["npz_path"])
    H_log = data[cfg["data"]["key"]]

    input_didxs = get_sparse_set(cfg["data"]["n_measurements"])
    D = len(input_didxs)

    unseen_didxs = sorted(set(range(cfg["data"]["n_dirs"])) - set(input_didxs))

    n_train = cfg["data"]["train_subjects"]

    train_subjects = np.arange(0, n_train)
    test_subjects = np.arange(n_train, cfg["data"]["n_subjects"])

    train_ds = SonicomHLogDataset(H_log, input_didxs, train_subjects)
    test_ds = SonicomHLogDataset(H_log, input_didxs, test_subjects)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
    )

    # -------------------------
    # Build model
    # -------------------------
    model = build_model(cfg, D).to(device)

    # -------------------------
    # Load checkpoint (PyTorch 2.6 safe)
    # -------------------------
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    # if checkpoint is pure state_dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    print("Loaded checkpoint:", ckpt_path)

    # -------------------------
    # Evaluate
    # -------------------------
    train_lsd, train_ild = evaluate(
        model, train_loader, unseen_didxs, device
    )

    test_lsd, test_ild = evaluate(
        model, test_loader, unseen_didxs, device
    )

    # -------------------------
    # Print results
    # -------------------------
    print("\n==============================")
    print("Evaluation Results")
    print("==============================")
    print(f"Sparse measurements : {cfg['data']['n_measurements']}")
    print(f"Train LSD (unseen)  : {train_lsd:.4f} dB")
    print(f"Test  LSD (unseen)  : {test_lsd:.4f} dB")
    print(f"Train ILD MAE       : {train_ild:.4f} dB")
    print(f"Test  ILD MAE       : {test_ild:.4f} dB")
    print("==============================\n")


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)

    args = parser.parse_args()

    main(args.config, args.ckpt)
