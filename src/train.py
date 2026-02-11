# src/train.py

import os
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from utils.seed import seed_everything
from utils.metrics import hrtf_shape_loss, lsd_loss_unseen
from datasets.sonicom_dataset import SonicomHLogDataset
from datasets.sparse_sets import get_sparse_set
from models.hrtf_upsampler import HRTFMagUpsampler


def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    seed_everything(cfg["seed"])

    device = torch.device(
        "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print("Using device:", device)

    # ==============================
    # Data
    # ==============================
    npz_path = cfg["data"]["npz_path"]
    data = np.load(npz_path)
    H_log = data[cfg["data"]["key"]]  # (200,793,2,106)

    input_didxs = get_sparse_set(cfg["data"]["n_measurements"])
    D = len(input_didxs)
    unseen_didxs = sorted(set(range(cfg["data"]["n_dirs"])) - set(input_didxs))

    print(f"Sparse measurements = {cfg['data']['n_measurements']}")
    print(f"D = {D}, unseen = {len(unseen_didxs)}")

    n_train = cfg["data"]["train_subjects"]
    train_subjects = np.arange(0, n_train)
    test_subjects = np.arange(n_train, cfg["data"]["n_subjects"])

    train_ds = SonicomHLogDataset(H_log, input_didxs, train_subjects)
    test_ds = SonicomHLogDataset(H_log, input_didxs, test_subjects)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # ==============================
    # Model
    # ==============================
    mcfg = cfg["model"]

    model = HRTFMagUpsampler(
        n_sparse=D,
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
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    # ==============================
    # Save dir with timestamp
    # ==============================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(cfg["train"]["save_dir"]) / f"{cfg['train']['exp_name']}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = save_dir / "best_model.pt"

    best_train_loss = float("inf")

    print("Start training...")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss_sum = 0.0

        for x_sparse, y_dense in train_loader:
            x_sparse = x_sparse.to(device)
            y_dense = y_dense.to(device)

            pred = model(x_sparse)
            loss = hrtf_shape_loss(
                pred,
                y_dense,
                lambda_grad=cfg["train"]["lambda_grad"],
                grad_type=cfg["train"]["grad_type"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            torch.save(model.state_dict(), best_ckpt_path)


            print(f"✅ Saved best model at epoch {epoch}")
            print(f"   Train loss = {best_train_loss:.6f}")
            print(f"   Path = {best_ckpt_path}")

        if epoch % cfg["train"]["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                test_lsd = 0.0
                for x_sparse, y_dense in test_loader:
                    x_sparse = x_sparse.to(device)
                    y_dense = y_dense.to(device)
                    pred = model(x_sparse)
                    test_lsd += lsd_loss_unseen(pred, y_dense, unseen_didxs).item()

                test_lsd /= len(test_loader)

            print(f"Epoch {epoch:04d} | Test LSD(unseen) = {test_lsd:.4f}")

    print("Training finished.")
    print("Best train loss:", best_train_loss)
    print("Best model saved at:", best_ckpt_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    main(args.config)
