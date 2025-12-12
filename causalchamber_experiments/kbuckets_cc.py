#!/usr/bin/env python3
"""
k-bucket causalpfn exp
median split inside each bucket
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from causalpfn import CATEEstimator, ATEEstimator
from causalchamber.datasets import Dataset as CCDataset


# config
DATA_ROOT = Path("./data")
OUT_BASE = Path("./results/lt_camera_causalpfn/kbuckets_median")

MAX_SAMPLES = 10000
SEED = 9

K = 6
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


# dino embeds
@torch.no_grad()
def compute_embeddings(model, imgs_uint8, transform, device, batch_size=64):
    ds = ImageDataset(imgs_uint8, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    out = []
    for batch in dl:
        feats = model.get_intermediate_layers(batch.to(device), n=1)[0][:, 0]
        out.append(feats.cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)


# load dino
def load_local_dinov3(device):
    print("load dino", flush=True)

    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=False)
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


# one bucket
def run_bucket(
    k,
    X,
    delta,
    Y,
    tau_true_raw,
    tau_true_std,
    bucket_mask,
    out_dir,
    device,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # slice
    Xk = X[bucket_mask]
    delt_k = delta[bucket_mask]
    Yk = Y[bucket_mask]
    tau_std_k = tau_true_std[bucket_mask]
    tau_raw_k = tau_true_raw[bucket_mask]

    # median split
    med_k = np.median(delt_k)
    Tk = (delt_k > med_k).astype(np.float32)

    with open(out_dir / "median_threshold.txt", "w") as f:
        f.write(f"bucket {k} median delta = {med_k:.6f}\n")

    # train/test
    n = len(Xk)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED + k)
    rng.shuffle(idx)

    split = int(0.7 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]

    Xtr, Xte = Xk[train_idx], Xk[test_idx]
    Ttr, Tte = Tk[train_idx], Tk[test_idx]
    Ytr, Yte = Yk[train_idx], Yk[test_idx]

    tau_te = tau_std_k[test_idx]
    delta_te = delt_k[test_idx]

    # cate
    cate = CATEEstimator(device=device)
    cate.fit(Xtr, Ttr, Ytr)
    cate_hat = cate.estimate_cate(Xte)

    # ate
    ate = ATEEstimator(device=device)
    ate.fit(Xk, Tk, Yk)
    ate_hat = float(ate.estimate_ate())

    true_ate = float(tau_raw_k[Tk == 1].mean() - tau_raw_k[Tk == 0].mean())
    pehe = float(np.sqrt(np.mean((cate_hat - tau_te) ** 2)))

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"bucket {k}\n")
        f.write(f"median delta = {med_k:.6f}\n")
        f.write(f"true ate raw = {true_ate:.6f}\n")
        f.write(f"ate hat = {ate_hat:.6f}\n")
        f.write(f"ate err = {abs(ate_hat - true_ate):.6f}\n")
        f.write(f"pehe = {pehe:.6f}\n")

    pd.DataFrame({
        "delta": delta_te,
        "T_k": Tte,
        "Y": Yte,
        "tau_true_std": tau_te,
        "cate_hat": cate_hat,
    }).to_csv(out_dir / "test_results.csv", index=False)

    # cate scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(tau_te, cate_hat, s=10, alpha=0.4)
    low = min(tau_te.min(), cate_hat.min())
    high = max(tau_te.max(), cate_hat.max())
    plt.plot([low, high], [low, high], "k--")
    plt.tight_layout()
    plt.savefig(out_dir / f"cate_scatter_bucket{k}.png", dpi=200)
    plt.close()

    # effects vs delta
    plt.figure(figsize=(6, 4))
    plt.scatter(delta_te, tau_te, s=10, alpha=0.3, label="true")
    plt.scatter(delta_te, cate_hat, s=10, alpha=0.3, label="hat")
    plt.axvline(med_k, linestyle="--", color="k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"effects_vs_delta_bucket{k}.png", dpi=200)
    plt.close()

    print("done bucket", k, flush=True)
    return cate_hat, tau_te, delta_te, ate_hat, true_ate, med_k


# combined plots
def plot_combined(
    K,
    all_delta,
    all_tau,
    all_hat,
    tau_true_raw,
    delta,
    ATE_hat_list,
    ATE_true_list,
    out_dir,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # true curve
    order = np.argsort(delta)
    plt.figure(figsize=(8, 4))
    plt.plot(delta[order], tau_true_raw[order], "b-")
    plt.tight_layout()
    plt.savefig(out_dir / "true_effect_full_curve.png", dpi=200)
    plt.close()

    # cate curves
    plt.figure(figsize=(8, 6))
    for k in range(K):
        o = np.argsort(all_delta[k])
        plt.plot(all_delta[k][o], all_hat[k][o], ".", markersize=4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "all_buckets_CATE.png", dpi=200)
    plt.close()

    # overlay
    order = np.argsort(delta)
    plt.figure(figsize=(8, 6))
    plt.plot(delta[order], tau_true_raw[order], "b-", linewidth=2)
    for k in range(K):
        o = np.argsort(all_delta[k])
        plt.plot(all_delta[k][o], all_hat[k][o], ".", alpha=0.4, markersize=4)
    plt.tight_layout()
    plt.savefig(out_dir / "true_vs_CATE_all_buckets.png", dpi=200)
    plt.close()

    # ate per bucket
    plt.figure(figsize=(6, 4))
    plt.plot(range(K), ATE_true_list, "bo-")
    plt.plot(range(K), ATE_hat_list, "ro-")
    plt.tight_layout()
    plt.savefig(out_dir / "ATE_comparison_buckets.png", dpi=200)
    plt.close()


# main
def main():
    set_all_seeds(SEED)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    ds = CCDataset("lt_camera_v1", root=DATA_ROOT, download=True)
    exp = ds.get_experiment("scm_1_pol_1")

    df = exp.as_pandas_dataframe()
    imgs = exp.as_image_array(size=IMG_SIZE)

    df = df.iloc[:MAX_SAMPLES].reset_index(drop=True)
    imgs = imgs[:MAX_SAMPLES]

    theta1 = df["pol_1"].to_numpy()
    theta2 = df["pol_2"].to_numpy()
    delta = theta2 - theta1

    Y_raw = df["brightness_value"].to_numpy()
    Y = (Y_raw - Y_raw.mean()) / Y_raw.std()

    tau_true_raw = np.cos(np.radians(delta)) ** 2
    tau_true_std = (tau_true_raw - tau_true_raw.mean()) / tau_true_raw.std()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm = load_local_dinov3(device)
    X = compute_embeddings(model, imgs, tfm, device)

    # add angles
    X_aug = np.concatenate([X, theta1[:, None], theta2[:, None]], axis=1)

    # bucket ids
    bins = np.linspace(delta.min(), delta.max(), K + 1)
    bucket_ids = np.digitize(delta, bins) - 1

    all_delta = {}
    all_tau = {}
    all_hat = {}
    ATE_true_list = []
    ATE_hat_list = []

    for k in range(K):
        mask = bucket_ids == k
        bdir = OUT_BASE / f"bucket_{k}"

        cate_hat, tau_k, delt_k, ate_hat, ate_true, med_k = run_bucket(
            k,
            X_aug,
            delta,
            Y,
            tau_true_raw,
            tau_true_std,
            mask,
            bdir,
            device,
        )

        all_delta[k] = delt_k
        all_tau[k] = tau_k
        all_hat[k] = cate_hat
        ATE_true_list.append(ate_true)
        ATE_hat_list.append(ate_hat)

    plot_combined(
        K,
        all_delta,
        all_tau,
        all_hat,
        tau_true_raw,
        delta,
        ATE_hat_list,
        ATE_true_list,
        OUT_BASE,
    )

    print("done", flush=True)


if __name__ == "__main__":
    main()
