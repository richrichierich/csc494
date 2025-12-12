#!/usr/bin/env python3
"""
synthpfn_embedding_analysis.py

quick probe + corr on dino embeds (synthetic mnist causal)
per digit: predict t, predict y, corr dims w/ t/y/z
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# paths
OUT_DIR = Path("./mnist_causal/mnist_causal/analysis")
EMB_PATH = OUT_DIR / "emb_train.npy"
META_PATH = OUT_DIR / "meta_train.csv"

RESULTS_DIR = OUT_DIR / "embedding_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# load
print("loading embeds + metadata...", flush=True)
E = np.load(EMB_PATH)
M = pd.read_csv(META_PATH)

assert len(E) == len(M), "embeddings and metadata don't match in length"
print(f"n={len(E)} d={E.shape[1]}", flush=True)
print("cols", M.columns.tolist(), flush=True)


digits = sorted(M["digit_label"].unique())
summary_rows = []

for d in digits:
    print(f"\ndigit {d}", flush=True)

    mask = (M["digit_label"] == d).to_numpy()
    Ed = E[mask]
    Md = M.loc[mask].reset_index(drop=True)

    # quick sanity: can a linear model read out t and y from embeddings?
    clf_T = LogisticRegression(max_iter=2000)
    acc_T = cross_val_score(clf_T, Ed, Md["T"], cv=5).mean()

    reg_Y = LinearRegression()
    r2_Y = cross_val_score(reg_Y, Ed, Md["Y"], cv=5, scoring="r2").mean()

    print(f"acc_T={acc_T:.3f} r2_Y={r2_Y:.3f}", flush=True)

    # per-dim pearson corr with t / y / z
    corr_T = np.zeros(Ed.shape[1], dtype=np.float32)
    corr_Y = np.zeros(Ed.shape[1], dtype=np.float32)
    corr_z = np.zeros(Ed.shape[1], dtype=np.float32)

    t = Md["T"].to_numpy()
    y = Md["Y"].to_numpy()
    z = Md["z_confounder"].to_numpy()

    for j in range(Ed.shape[1]):
        corr_T[j] = pearsonr(Ed[:, j], t)[0]
        corr_Y[j] = pearsonr(Ed[:, j], y)[0]
        corr_z[j] = pearsonr(Ed[:, j], z)[0]

    corr_df = pd.DataFrame(
        {
            "dim": np.arange(Ed.shape[1]),
            "corr_T": corr_T,
            "corr_Y": corr_Y,
            "corr_z": corr_z,
        }
    )
    corr_df.to_csv(RESULTS_DIR / f"digit{d}_corrs.csv", index=False)

    # top dims by abs corr (for quick inspection)
    top_T = np.argsort(np.abs(corr_T))[-5:]
    top_Y = np.argsort(np.abs(corr_Y))[-5:]
    top_z = np.argsort(np.abs(corr_z))[-5:]

    print("top_t", top_T.tolist(), flush=True)
    print("top_y", top_Y.tolist(), flush=True)
    print("top_z", top_z.tolist(), flush=True)

    summary_rows.append(
        {
            "digit": int(d),
            "logreg_acc_T": float(acc_T),
            "linreg_r2_Y": float(r2_Y),
            "max_abs_corr_T": float(np.max(np.abs(corr_T))),
            "max_abs_corr_Y": float(np.max(np.abs(corr_Y))),
            "max_abs_corr_z": float(np.max(np.abs(corr_z))),
        }
    )


# save
summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_DIR / "embedding_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nsaved", summary_path, flush=True)
print(summary_df.round(3).to_string(index=False), flush=True)
