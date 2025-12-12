#!/usr/bin/env python3
"""
exp 1: lt_camera_v1 + dinov3 + causalpfn

T = 1(pol_2 > median(pol_2))
Y = normalized brightness
tau = cos^2(pol_2 - pol_1)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from causalchamber.datasets import Dataset as CCDataset
from causalpfn import CATEEstimator, ATEEstimator


# config
DATA_ROOT = Path("./data")
OUT_BASE = Path("./results_pol2T")

MAX_SAMPLES = 10_000
SEED = 9
IMG_SIZE = "100"

REPO_DIR = os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3")
WEIGHTS = os.path.expanduser(
    "~/projects/aip-rahulgk/richguo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)


# seeds
def set_all_seeds(seed=9):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# image ds
class ImageDataset(Dataset):
    # tiny wrapper
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx])


# dino embed
@torch.no_grad()
def compute_embeddings(model, imgs, transform, device, bs=64):
    ds = ImageDataset(imgs, transform)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)

    out = []
    for i, batch in enumerate(dl):
        batch = batch.to(device)
        feats = model.get_intermediate_layers(batch, n=1)[0][:, 0]
        out.append(feats.cpu().numpy())

        if i == 0 or (i + 1) % 10 == 0:
            print(f"embed {i+1}/{len(dl)}", flush=True)

    return np.concatenate(out, axis=0).astype(np.float32)


# load dino
def load_local_dinov3(device):
    print("load dino", flush=True)

    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model, tfm


# sanity plots
def generate_diagnostics(delta, tau_raw, tau_std, Y, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=80)
    plt.tight_layout()
    plt.savefig(out_dir / "dist_delta_theta.png", dpi=200)
    plt.close()

    order = np.argsort(delta)
    plt.figure(figsize=(6, 4))
    plt.plot(delta[order], tau_raw[order], ".", alpha=0.5, markersize=3)
    plt.tight_layout()
    plt.savefig(out_dir / "true_effect_cos2.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta, tau_std, s=4, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tau_standardized_vs_delta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta, Y, s=4, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "Y_vs_delta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(tau_std, Y, s=4, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "Y_vs_tau_true.png", dpi=200)
    plt.close()


# run once
def run_experiment(
    X,
    theta1,
    theta2,
    delta,
    T,
    Y,
    tau_raw,
    tau_std,
    tag,
    out_dir,
    device,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("run", tag, flush=True)

    n = len(X)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)

    split = int(0.7 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    Xtr, Xte = X[train_idx], X[test_idx]
    Ttr, Tte = T[train_idx], T[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]

    delta_te = delta[test_idx]
    tau_te_std = tau_std[test_idx]
    tau_te_raw = tau_raw[test_idx]

    cate = CATEEstimator(device=device)
    cate.fit(Xtr, Ttr, Ytr)
    cate_hat = cate.estimate_cate(Xte)

    ate = ATEEstimator(device=device)
    ate.fit(X, T, Y)
    ate_hat = float(ate.estimate_ate())

    ate_true_raw = float(tau_raw[T == 1].mean() - tau_raw[T == 0].mean())
    ate_true_std = float(tau_std[T == 1].mean() - tau_std[T == 0].mean())
    pehe = float(np.sqrt(np.mean((cate_hat - tau_te_std) ** 2)))

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"tag: {tag}\n")
        f.write(f"true ate raw: {ate_true_raw}\n")
        f.write(f"true ate std: {ate_true_std}\n")
        f.write(f"est ate: {ate_hat}\n")
        f.write(f"pehe: {pehe}\n")

    pd.DataFrame({
        "pol_1": theta1[test_idx],
        "pol_2": theta2[test_idx],
        "delta": delta_te,
        "T": Tte,
        "Y": Yte,
        "tau_true_raw": tau_te_raw,
        "tau_true_std": tau_te_std,
        "cate_hat": cate_hat,
    }).to_csv(out_dir / "test_results.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(tau_te_std, cate_hat, s=10, alpha=0.4)
    low = min(tau_te_std.min(), cate_hat.min())
    high = max(tau_te_std.max(), cate_hat.max())
    plt.plot([low, high], [low, high], "k--")
    plt.tight_layout()
    plt.savefig(out_dir / f"cate_scatter_{tag}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta_te, cate_hat, s=10, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_vs_cate_hat_{tag}.png", dpi=200)
    plt.close()

    print("done", tag, flush=True)


# main
def main():
    set_all_seeds(SEED)

    DATA_ROOT.mkdir(exist_ok=True)
    OUT_BASE.mkdir(exist_ok=True)

    ds = CCDataset("lt_camera_v1", root=DATA_ROOT, download=True)
    exp = ds.get_experiment("scm_1_pol_1")

    df = exp.as_pandas_dataframe()
    imgs = exp.as_image_array(size=IMG_SIZE)

    n = min(MAX_SAMPLES, len(df))
    df = df.iloc[:n].reset_index(drop=True)
    imgs = imgs[:n]

    theta1 = df["pol_1"].to_numpy()
    theta2 = df["pol_2"].to_numpy()
    delta = theta2 - theta1

    Y_raw = df["brightness_value"].to_numpy()
    Y = (Y_raw - Y_raw.mean()) / Y_raw.std()

    tau_raw = np.cos(np.radians(delta)) ** 2
    tau_std = (tau_raw - tau_raw.mean()) / tau_raw.std()

    T = (theta2 > np.median(theta2)).astype(np.float32)

    generate_diagnostics(delta, tau_raw, tau_std, Y, OUT_BASE / "diagnostics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm = load_local_dinov3(device)
    X_dino = compute_embeddings(model, imgs, tfm, device)

    run_experiment(
        X_dino,
        theta1,
        theta2,
        delta,
        T,
        Y,
        tau_raw,
        tau_std,
        "base",
        OUT_BASE / "base",
        device,
    )

    X_aug = np.concatenate([X_dino, theta1[:, None], theta2[:, None]], axis=1)

    run_experiment(
        X_aug,
        theta1,
        theta2,
        delta,
        T,
        Y,
        tau_raw,
        tau_std,
        "augmented",
        OUT_BASE / "augmented",
        device,
    )

    print("done", flush=True)


if __name__ == "__main__":
    main()
