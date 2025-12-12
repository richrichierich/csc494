#!/usr/bin/env python3
"""
run_causalpfn_mnist_ood.py

ood generalization test
train on digits 0-4, evaluate on digits 5-9
reads ./mnist_causal_benchmark/analysis_hetero/
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
OUT_DIR = DATA_DIR / "causalpfn_ood_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("loading...", flush=True)
E = np.load(DATA_DIR / "emb_train.npy")
M = pd.read_csv(DATA_DIR / "meta_train.csv")

print(f"n={len(M)} d={E.shape[1]}", flush=True)
print("cols", list(M.columns), flush=True)


# prep arrays
D = M["digit_label"].astype(int).to_numpy()
T = M["T"].astype(float).to_numpy().astype(np.float32)
Y = M["Y"].astype(float).to_numpy().astype(np.float32)

train_mask = np.isin(D, [0, 1, 2, 3, 4])
test_mask = np.isin(D, [5, 6, 7, 8, 9])

X_train = E[train_mask]
T_train = T[train_mask]
Y_train = Y[train_mask]
D_train = D[train_mask]

X_test = E[test_mask]
T_test = T[test_mask]
Y_test = Y[test_mask]
D_test = D[test_mask]

print(f"train digits={sorted(set(D_train))} n={len(D_train)}", flush=True)
print(f"test digits={sorted(set(D_test))} n={len(D_test)}", flush=True)


# scale features (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)


# fit on a stratified split of the id set (so we have a quick id sanity check)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=9)
tr_idx, val_idx = next(splitter.split(X_train, T_train))

print("fitting...", flush=True)
cate_est = CATEEstimator(device=device, verbose=False)
ate_est = ATEEstimator(device=device, verbose=False)

cate_est.fit(X_train[tr_idx], T_train[tr_idx], Y_train[tr_idx])
ate_est.fit(X_train[tr_idx], T_train[tr_idx], Y_train[tr_idx])


# eval id
print("eval id...", flush=True)
cate_val = cate_est.estimate_cate(X_train[val_idx])

# note: for ate, the estimator usually returns the trained ate; we report it once
ate_val = ate_est.estimate_ate()

print(f"id cate={cate_val.mean():.3f} ±{cate_val.std():.3f} ate={ate_val:.3f}", flush=True)


# eval ood
print("eval ood...", flush=True)
cate_ood = cate_est.estimate_cate(X_test)
ate_ood = ate_est.estimate_ate()

print(f"ood cate={cate_ood.mean():.3f} ±{cate_ood.std():.3f} ate={ate_ood:.3f}", flush=True)


# per-digit breakdown on ood set
ood_rows = []
for d in sorted(set(D_test)):
    m = (D_test == d)

    cate_d = cate_est.estimate_cate(X_test[m])

    # this api call exists in some versions; if yours errors, swap to fitting per digit
    ate_d = ate_est.estimate_ate(X_test[m], T_test[m], Y_test[m])

    ood_rows.append(
        {
            "digit": int(d),
            "n": int(m.sum()),
            "CATE_mean": float(cate_d.mean()),
            "CATE_std": float(cate_d.std()),
            "ATE_hat": float(ate_d),
        }
    )

    print(f"d={d} cate={cate_d.mean():.3f} ±{cate_d.std():.3f} ate={ate_d:.3f}", flush=True)


# save
out_csv = OUT_DIR / "causalpfn_ood_estimates.csv"
pd.DataFrame(ood_rows).to_csv(out_csv, index=False)

summary = {
    "IND_CATE_mean": float(cate_val.mean()),
    "IND_CATE_std": float(cate_val.std()),
    "IND_ATE": float(ate_val),
    "OOD_CATE_mean": float(cate_ood.mean()),
    "OOD_CATE_std": float(cate_ood.std()),
    "OOD_ATE": float(ate_ood),
}
pd.Series(summary).to_csv(OUT_DIR / "summary.csv")

print("saved", out_csv.resolve(), flush=True)
print("summary", summary, flush=True)
