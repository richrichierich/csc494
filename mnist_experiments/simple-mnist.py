#!/usr/bin/env python3
"""
synthpfn_generate_xy_causal_hetero.py

mnist causal gen
D -> T -> Y, plus D -> Y via f(T,D)
also saves gt effects
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms


#setup
RNG_SEED = 9
TRAIN_N = 20000
TEST_N = 1000
Y_NOISE_STD = 0.2
IMG_SIZE = 28 * 28

OUT_ROOT = Path("./mnist_causal_benchmark")
ANALYSIS_DIR = OUT_ROOT / "analysis_hetero"
GT_DIR = OUT_ROOT / "ground_truth"
for d in [ANALYSIS_DIR, GT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPO_DIR = os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3")
WEIGHTS = os.path.expanduser(
    "~/projects/aip-rahulgk/richguo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

EPS = 1e-8


#seeds
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


#calibrate s_d
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


#mnist subset
def load_mnist_subset(train_n, test_n):
    transform = transforms.ToTensor()
    data_root = Path("./data")
    data_root.mkdir(parents=True, exist_ok=True)

    full_train = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(RNG_SEED)
    idx_train = torch.randperm(len(full_train), generator=g)[:train_n].tolist()
    idx_test = torch.randperm(len(full_test), generator=g)[:test_n].tolist()

    sub_train = torch.utils.data.Subset(full_train, idx_train)
    sub_test = torch.utils.data.Subset(full_test, idx_test)
    return sub_train, sub_test


#to numpy
def mnist_to_numpy(ds):
    imgs, labels = [], []
    for img, label in ds:
        imgs.append(img.numpy().squeeze())
        labels.append(label)
    X = np.stack(imgs, axis=0)
    y = np.array(labels)
    return X, y


#treat params
def sample_class_params(D_in, seed=0):
    rng = np.random.default_rng(seed)
    w_mat, b_vec = [], []
    for d in range(10):
        mean_shift = np.sin(2 * np.pi * d / 10)
        w_d = rng.normal(loc=mean_shift * 0.25, scale=0.35, size=(D_in,))
        b_d = rng.normal(loc=0.35 * mean_shift, scale=0.25)
        w_mat.append(w_d)
        b_vec.append(b_d)
    return np.stack(w_mat), np.array(b_vec)


#f(T,D)
class HeteroOutcome(nn.Module):
    # per-digit nonlin
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

    def forward(self, onehot_D, T):
        h_d = self.f_base(onehot_D).squeeze(-1)
        g_d = self.f_treat(onehot_D).squeeze(-1)
        nonlinear = torch.sin(np.pi * T + 0.3 * onehot_D @ torch.arange(10.0, device=onehot_D.device))
        return h_d + g_d * T + nonlinear


#gen T,Y
def generate_T_Y_hetero(X, digits, w_mat, b_vec, sigma_y=0.2, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    Xf = X.reshape(N, -1)  # flat

    logits_raw = np.zeros(N)
    z_logits = np.zeros(N)
    p_T = np.zeros(N)
    T = np.zeros(N, dtype=np.int64)

    p_targets = np.linspace(0.25, 0.75, 10)
    for d in range(10):
        mask = digits == d
        if not np.any(mask):
            continue

        l = Xf[mask] @ w_mat[d] + b_vec[d]
        z = (l - l.mean()) / (l.std() + EPS)  # z
        s_d = find_shift_for_target(z, p_targets[d])
        p = sigmoid(z + s_d)

        logits_raw[mask] = l
        z_logits[mask] = z
        p_T[mask] = p
        T[mask] = rng.binomial(1, p)

    onehot = np.zeros((N, 10), dtype=np.float32)
    onehot[np.arange(N), digits] = 1.0

    model = HeteroOutcome(hidden=32)
    with torch.no_grad():
        Y = model(torch.tensor(onehot), torch.tensor(T, dtype=torch.float32)).numpy()
    Y += rng.normal(0, sigma_y, size=N)  # noise

    return {
        "T": T,
        "Y": Y.astype(np.float32),
        "p_T": p_T.astype(np.float32),
        "logits_raw": logits_raw.astype(np.float32),
        "z_logits": z_logits.astype(np.float32),
        "p_targets": p_targets.tolist(),
        "model_state": model.state_dict(),
    }


#load dino
def load_dino():
    print("load dino", flush=True)
    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device={device}", flush=True)
    return model, device


#dino embeds
@torch.no_grad()
def compute_embeddings(model, device, imgs_np):
    transform = transforms.Compose(
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

    E = []
    for img_np in imgs_np:
        t = transform(img_np).unsqueeze(0).to(device)
        out = model.get_intermediate_layers(t, n=1)[0]
        cls = out[:, 0].cpu().numpy()  # cls
        E.append(cls)

    return np.concatenate(E, axis=0)


#gt effects
def compute_ground_truth(df: pd.DataFrame, out_path: Path):
    print("gt", flush=True)

    def tau(g):
        if (g["T"] == 1).sum() == 0 or (g["T"] == 0).sum() == 0:
            return np.nan
        return g[g.T == 1]["Y"].mean() - g[g.T == 0]["Y"].mean()

    tau_d = df.groupby("digit_label").apply(tau)
    ate_true = df[df.T == 1]["Y"].mean() - df[df.T == 0]["Y"].mean()

    gt_df = pd.DataFrame({"digit_label": tau_d.index, "tau_true": tau_d.values})
    gt_df.loc[len(gt_df)] = ["GLOBAL_ATE", ate_true]
    gt_df.to_csv(out_path, index=False)

    print(gt_df.to_string(index=False), flush=True)
    print("saved", out_path, flush=True)


#main
def main():
    set_all_seeds(RNG_SEED)

    print("load mnist", flush=True)
    tr_ds, te_ds = load_mnist_subset(TRAIN_N, TEST_N)
    Xtr, Dtr = mnist_to_numpy(tr_ds)
    Xte, Dte = mnist_to_numpy(te_ds)

    print("sample params", flush=True)
    w_mat, b_vec = sample_class_params(IMG_SIZE, seed=RNG_SEED + 10)

    print("gen T,Y", flush=True)
    out_tr = generate_T_Y_hetero(Xtr, Dtr, w_mat, b_vec, sigma_y=Y_NOISE_STD, seed=RNG_SEED + 1)
    out_te = generate_T_Y_hetero(Xte, Dte, w_mat, b_vec, sigma_y=Y_NOISE_STD, seed=RNG_SEED + 2)

    print("embed", flush=True)
    model, device = load_dino()
    E_tr = compute_embeddings(model, device, Xtr)
    E_te = compute_embeddings(model, device, Xte)

    print("save", flush=True)
    np.save(ANALYSIS_DIR / "emb_train.npy", E_tr)
    np.save(ANALYSIS_DIR / "emb_test.npy", E_te)

    meta_train = pd.DataFrame(
        {"digit_label": Dtr, "T": out_tr["T"], "Y": out_tr["Y"], "p_T": out_tr["p_T"]}
    )
    meta_test = pd.DataFrame(
        {"digit_label": Dte, "T": out_te["T"], "Y": out_te["Y"], "p_T": out_te["p_T"]}
    )
    meta_train.to_csv(ANALYSIS_DIR / "meta_train.csv", index=False)
    meta_test.to_csv(ANALYSIS_DIR / "meta_test.csv", index=False)

    torch.save(out_tr["model_state"], ANALYSIS_DIR / "hetero_outcome_model.pt")
    with open(ANALYSIS_DIR / "generator_config.json", "w") as f:
        json.dump(
            {
                "seed": RNG_SEED,
                "y_noise_std": Y_NOISE_STD,
                "note": "nonlinear f(T,D)+eps with heterogeneous treatment effects per digit",
            },
            f,
            indent=2,
        )

    compute_ground_truth(meta_train, GT_DIR / "true_effects.csv")

    print("done", flush=True)
    print(ANALYSIS_DIR, flush=True)
    print(GT_DIR, flush=True)


if __name__ == "__main__":
    main()
