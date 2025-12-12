#!/usr/bin/env python3
"""
lt_camera + dinov3 + causalpfn

runs two versions:
- base dino
- dino + (pol1, pol2)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from causalpfn import CATEEstimator, ATEEstimator
from causalchamber.datasets import Dataset as CCDataset


# config
DATA_ROOT = Path("./data")
OUT_BASE = Path("./results/lt_camera_causalpfn")
DIAG_DIR = OUT_BASE / "diagnostics"

MAX_SAMPLES = 10000
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


# image wrapper
class ImageDataset(Dataset):
    # simple wrapper
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx])


# dino embeddings
@torch.no_grad()
def compute_embeddings(model, imgs_uint8, transform, device, batch_size=64):
    ds = ImageDataset(imgs_uint8, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    embeds = []
    for i, batch in enumerate(dl):
        batch = batch.to(device)
        feats = model.get_intermediate_layers(batch, n=1)[0][:, 0]
        embeds.append(feats.cpu().numpy())

        if i == 0 or (i + 1) % 10 == 0:
            print(f"embed {i+1}/{len(dl)}", flush=True)

    return np.concatenate(embeds, axis=0).astype(np.float32)


# load dino
def load_local_dinov3(device):
    print("load dino", flush=True)

    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model, transform


# sanity plots
def generate_diagnostics(delta, tau_raw, tau_std, Y, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=80, alpha=0.7)
    plt.xlabel("Δθ")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "dist_delta_theta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta, tau_raw, s=4, alpha=0.3)
    plt.xlabel("Δθ")
    plt.ylabel("cos²(Δθ)")
    plt.tight_layout()
    plt.savefig(out_dir / "true_effect_cos2.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta, tau_std, s=4, alpha=0.3)
    plt.xlabel("Δθ")
    plt.ylabel("τ_true")
    plt.tight_layout()
    plt.savefig(out_dir / "tau_standardized_vs_delta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(delta, Y, s=3, alpha=0.3)
    plt.xlabel("Δθ")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(out_dir / "Y_vs_delta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(tau_std, Y, s=4, alpha=0.3)
    plt.xlabel("τ_true")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(out_dir / "Y_vs_tau_true.png", dpi=200)
    plt.close()


# one run
def run_experiment(X, T, Y, tau_raw, tau_std, delta, tag, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("run", tag, flush=True)

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    split = int(0.7 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]

    Xtr, Xte = X[train_idx], X[test_idx]
    Ttr, Tte = T[train_idx], T[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]
    tau_te = tau_std[test_idx]
    delta_te = delta[test_idx]

    cate_est = CATEEstimator(device=device)
    cate_est.fit(Xtr, Ttr, Ytr)
    cate_hat = cate_est.estimate_cate(Xte)

    ate_est = ATEEstimator(device=device)
    ate_est.fit(X, T, Y)
    ate_hat = float(ate_est.estimate_ate())

    ate_true = float(tau_raw[T == 1].mean() - tau_raw[T == 0].mean())
    pehe = float(np.sqrt(np.mean((cate_hat - tau_te) ** 2)))

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"ATE_true = {ate_true:.6f}\n")
        f.write(f"ATE_hat  = {ate_hat:.6f}\n")
        f.write(f"ATE_err  = {abs(ate_hat - ate_true):.6f}\n")
        f.write(f"PEHE     = {pehe:.6f}\n")

    pd.DataFrame({
        "delta": delta_te,
        "T": Tte,
        "Y": Yte,
        "tau_true_raw": tau_raw[test_idx],
        "tau_true_std": tau_te,
        "cate_hat": cate_hat,
    }).to_csv(out_dir / "test_results.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(tau_te, cate_hat, s=10, alpha=0.4)
    low = min(tau_te.min(), cate_hat.min())
    high = max(tau_te.max(), cate_hat.max())
    plt.plot([low, high], [low, high], "k--")
    plt.xlabel("τ_true")
    plt.ylabel("cate_hat")
    plt.tight_layout()
    plt.savefig(out_dir / f"cate_scatter_{tag}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(delta_te, tau_te, s=8, alpha=0.35, label="τ_true")
    plt.scatter(delta_te, cate_hat, s=8, alpha=0.35, label="cate_hat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_vs_effects_{tag}.png", dpi=200)
    plt.close()

    print("done", tag, flush=True)


# main
def main():
    set_all_seeds(SEED)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    dataset = CCDataset("lt_camera_v1", root=DATA_ROOT, download=True)
    experiment = dataset.get_experiment("scm_1_pol_1")

    df = experiment.as_pandas_dataframe()
    imgs = experiment.as_image_array(size=IMG_SIZE)

    N = min(MAX_SAMPLES, len(df))
    df = df.iloc[:N].reset_index(drop=True)
    imgs = imgs[:N]

    theta1 = df["pol_1"].to_numpy()
    theta2 = df["pol_2"].to_numpy()
    delta = theta2 - theta1

    T = (delta > np.median(delta)).astype(np.float32)

    Y_raw = df["brightness_value"].to_numpy()
    Y = (Y_raw - Y_raw.mean()) / Y_raw.std()

    tau_raw = np.cos(np.radians(delta)) ** 2
    tau_std = (tau_raw - tau_raw.mean()) / tau_raw.std()

    generate_diagnostics(delta, tau_raw, tau_std, Y, DIAG_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_local_dinov3(device)
    X = compute_embeddings(model, imgs, transform, device)
    print("embeds", X.shape, flush=True)

    run_experiment(
        X, T, Y, tau_raw, tau_std, delta,
        tag="base",
        out_dir=OUT_BASE / "base",
        device=device,
    )

    X_aug = np.concatenate([X, theta1[:, None], theta2[:, None]], axis=1)
    run_experiment(
        X_aug, T, Y, tau_raw, tau_std, delta,
        tag="augmented",
        out_dir=OUT_BASE / "augmented",
        device=device,
    )

    print("done", flush=True)


if __name__ == "__main__":
    main()
