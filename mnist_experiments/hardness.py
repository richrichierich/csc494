#!/usr/bin/env python3
"""
mnist conf sweep
add latent C(D) into T + Y
id=0-4, ood=5-9
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from causalpfn import CATEEstimator, ATEEstimator


device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = os.path.expanduser(
    "~/projects/aip-rahulgk/richguo/mnist_viz/hardness_causalpfn_conflevels"
)
os.makedirs(save_dir, exist_ok=True)

n_samples = 10000
y_noise_std = 0.2
eps = 1e-8

repo_dir = os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3")
weights = os.path.expanduser(
    "~/projects/aip-rahulgk/richguo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

hardness_levels = [
    {"name": "easy", "confounding": 0.1},
    {"name": "medium", "confounding": 0.5},
    {"name": "hard", "confounding": 0.9},
]


#load mnist+dino
@torch.no_grad()
def load_mnist_dino_embeddings(n_samples=10000):
    print("load mnist", flush=True)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(
        root="/home/richguo/projects/data",
        train=True,
        download=True,
        transform=transform,
    )
    imgs, labels = zip(*mnist)
    imgs, labels = imgs[:n_samples], np.array(labels[:n_samples])

    print("load dino", flush=True)
    model = torch.hub.load(repo_dir, "dinov3_vitl16", source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"device={device}", flush=True)

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    print("embed", flush=True)
    E = []
    for i, img in enumerate(imgs):
        t = preprocess(img).unsqueeze(0).to(device)
        out = model.get_intermediate_layers(t, n=1)[0]
        cls = out[:, 0].cpu().numpy()  # cls
        E.append(cls)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(imgs)}", flush=True)

    E = np.concatenate(E, axis=0)
    print(f"shape={E.shape}", flush=True)
    return E, labels


#sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


#calibrate shift
def find_shift_for_target(z_logits: np.ndarray, target: float, iters: int = 60, tol: float = 1e-5):
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = sigmoid(z_logits + mid).mean()
        if m > target:
            hi = mid
        else:
            lo = mid
        if abs(m - target) < tol:
            return mid
    return 0.5 * (lo + hi)


#f(T,D)
class HeteroOutcome(nn.Module):
    # per-digit mlps
    def __init__(self, hidden=32):
        super().__init__()
        self.f_base = nn.Sequential(
            nn.Linear(10, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.f_treat = nn.Sequential(
            nn.Linear(10, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, onehot_D, T_float):
        h_d = self.f_base(onehot_D).squeeze(-1)
        g_d = self.f_treat(onehot_D).squeeze(-1)
        idx = onehot_D @ torch.arange(10.0, device=onehot_D.device)  # digit idx
        nonlinear = torch.sin(np.pi * T_float + 0.3 * idx)
        return h_d + g_d * T_float + nonlinear


#sample T,Y
def generate_T_Y_with_confounding(D_std, labels, conf_level, y_noise_std=0.2, seed=9):
    # T uses + alpha*C(D)
    # Y uses + alpha*C(D) too
    print(f"alpha={conf_level}", flush=True)

    rng = np.random.default_rng(seed)
    N, p = D_std.shape

    w_c = rng.normal(0, 1, size=p)  # conf weights
    C = np.tanh(D_std @ w_c)  # conf

    T = np.zeros(N, dtype=np.int64)
    p_targets = np.linspace(0.25, 0.75, 10)

    for d in range(10):
        mask = labels == d
        if not np.any(mask):
            continue

        w_d = rng.normal(0, 1, size=p)
        b_d = rng.normal()
        l = D_std[mask] @ w_d + b_d
        z = (l - l.mean()) / (l.std() + eps)  # zscore
        s_d = find_shift_for_target(z, p_targets[d])

        logits = z + s_d + conf_level * C[mask]
        p_d = sigmoid(logits)
        T[mask] = rng.binomial(1, p_d)

    onehot = np.zeros((N, 10), dtype=np.float32)
    onehot[np.arange(N), labels] = 1.0

    model = HeteroOutcome(hidden=32)

    with torch.no_grad():
        onehot_t = torch.tensor(onehot)
        T_t = torch.tensor(T, dtype=torch.float32)
        fTD = model(onehot_t, T_t).numpy()
        f1 = model(onehot_t, torch.ones_like(T_t)).numpy()
        f0 = model(onehot_t, torch.zeros_like(T_t)).numpy()

    Y = fTD + conf_level * C + rng.normal(0, y_noise_std, size=N)  # add noise
    ATE_true = float(np.mean(f1 - f0))

    print(f"ate_true={ATE_true:.4f}", flush=True)
    return T.astype(np.float32), Y.astype(np.float32), ATE_true


#run pfn
def evaluate_causalpfn(X, T, Y, device="cpu"):
    print(f"pfn n={len(X)}", flush=True)

    if len(np.unique(T)) < 2:
        print("skip", flush=True)
        return np.nan, np.nan, np.nan

    cate_est = CATEEstimator(device=device, verbose=False)
    cate_est.fit(X, T, Y)
    cate_hat = cate_est.estimate_cate(X)
    cate_mean = float(np.mean(cate_hat))
    cate_std = float(np.std(cate_hat))

    ate_est = ATEEstimator(device=device, verbose=False)
    ate_est.fit(X, T, Y)
    ate_hat = float(ate_est.estimate_ate())

    print(f"ate={ate_hat:.4f} cate={cate_mean:.4f} std={cate_std:.4f}", flush=True)
    return ate_hat, cate_mean, cate_std


#main
def main():
    print(f"start device={device}", flush=True)

    D, labels = load_mnist_dino_embeddings(n_samples)

    print("scale", flush=True)
    X = StandardScaler().fit_transform(D).astype(np.float32)

    id_mask = np.isin(labels, [0, 1, 2, 3, 4])
    ood_mask = np.isin(labels, [5, 6, 7, 8, 9])
    print(f"id={id_mask.sum()} ood={ood_mask.sum()}", flush=True)

    results = []
    for cfg in hardness_levels:
        conf_level = cfg["confounding"]
        print(f"{cfg['name']} alpha={conf_level}", flush=True)

        for subset, mask in [("ID", id_mask), ("OOD", ood_mask)]:
            print(f"{subset} n={mask.sum()}", flush=True)

            X_sub = X[mask]
            labels_sub = labels[mask]

            T, Y, ate_true = generate_T_Y_with_confounding(
                X_sub, labels_sub, conf_level, y_noise_std=y_noise_std, seed=9
            )

            ate_hat, cate_mean, cate_std = evaluate_causalpfn(X_sub, T, Y, device=device)

            results.append(
                {
                    "hardness": cfg["name"],
                    "subset": subset,
                    "n": len(X_sub),
                    "ATE_true": ate_true,
                    "ATE_hat": ate_hat,
                    "CATE_mean": cate_mean,
                    "CATE_std": cate_std,
                }
            )

            print(
                f"{subset} ate_hat={ate_hat:.4f} ate_true={ate_true:.4f} "
                f"cate={cate_mean:.4f} std={cate_std:.4f}",
                flush=True,
            )

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "causalpfn_conflevels_results.csv")
    df.to_csv(csv_path, index=False)
    print("saved", csv_path, flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    hardness_names = [h["name"] for h in hardness_levels]

    # ate curve
    for subset in ["ID", "OOD"]:
        sub = df[df["subset"] == subset]
        axs[0].plot(hardness_names, sub["ATE_hat"], marker="o", label=f"{subset}-ate est")
        axs[0].plot(hardness_names, sub["ATE_true"], marker="x", linestyle="--", label=f"{subset}-ate true")
    axs[0].set_title("ate vs conf")
    axs[0].set_xlabel("conf")
    axs[0].set_ylabel("ate")
    axs[0].grid(True)
    axs[0].legend()

    # cate mean
    for subset in ["ID", "OOD"]:
        sub = df[df["subset"] == subset]
        axs[1].plot(hardness_names, sub["CATE_mean"], marker="s", label=f"{subset}-cate mean")
        axs[1].plot(hardness_names, sub["ATE_true"], marker="x", linestyle="--", label=f"{subset}-ate true ref")
    axs[1].set_title("cate mean vs conf")
    axs[1].set_xlabel("conf")
    axs[1].set_ylabel("mean cate")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "causalpfn_conflevels_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("saved", fig_path, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
