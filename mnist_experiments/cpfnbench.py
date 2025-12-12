#!/usr/bin/env python3
"""
causalpfn mnist benchmark

runs causalpfn on synthetic mnist causal data
per digit + all digits
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from causalpfn import ATEEstimator, CATEEstimator


# paths
DATA_DIR = Path("./mnist_causal_benchmark/analysis_hetero")
OUT_DIR = DATA_DIR / "causalpfn_results_hetero"
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("loading data...", flush=True)
E = np.load(DATA_DIR / "emb_train.npy")
M = pd.read_csv(DATA_DIR / "meta_train.csv")

print(f"n={len(M)} d={E.shape[1]}", flush=True)
print("cols", list(M.columns), flush=True)


# arrays
D = M["digit_label"].astype(int).to_numpy()
T = M["T"].astype(float).to_numpy().astype(np.float32)
Y = M["Y"].astype(float).to_numpy().astype(np.float32)

# scaling embeddings helps a bit for some models, and it's cheap
scaler = StandardScaler()
X = scaler.fit_transform(E).astype(np.float32)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)


def run_one_group(Xg: np.ndarray, Tg: np.ndarray, Yg: np.ndarray, tag: int, device: str) -> dict:
    # if the group is all treated or all control, causalpfn can't really do anything
    if len(np.unique(Tg)) < 2:
        print(f"skip group {tag} (degenerate T)", flush=True)
        return {"digit": tag, "n": len(Tg), "ATE_hat": np.nan, "CATE_mean": np.nan, "CATE_std": np.nan}

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9)
    tr_idx, te_idx = next(splitter.split(Xg, Tg))

    X_tr, X_te = Xg[tr_idx], Xg[te_idx]
    T_tr, T_te = Tg[tr_idx], Tg[te_idx]
    Y_tr, Y_te = Yg[tr_idx], Yg[te_idx]

    # cate on a held-out split
    cate_est = CATEEstimator(device=device, verbose=False)
    cate_est.fit(X_tr, T_tr, Y_tr)
    cate_hat = cate_est.estimate_cate(X_te)

    cate_mean = float(np.mean(cate_hat))
    cate_std = float(np.std(cate_hat))

    # ate on the whole group
    ate_est = ATEEstimator(device=device, verbose=False)
    ate_est.fit(Xg, Tg, Yg)
    ate_hat = float(ate_est.estimate_ate())

    return {"digit": tag, "n": len(Xg), "ATE_hat": ate_hat, "CATE_mean": cate_mean, "CATE_std": cate_std}


results = []

print("running per digit...", flush=True)
for digit in sorted(np.unique(D)):
    m = (D == digit)
    results.append(run_one_group(X[m], T[m], Y[m], tag=int(digit), device=device))

print("running all digits...", flush=True)
results.append(run_one_group(X, T, Y, tag=999, device=device))


df = pd.DataFrame(results, columns=["digit", "n", "ATE_hat", "CATE_mean", "CATE_std"])
out_csv = OUT_DIR / "causalpfn_estimates.csv"
df.to_csv(out_csv, index=False)

print("saved", out_csv.resolve(), flush=True)
print(df.to_string(index=False), flush=True)
