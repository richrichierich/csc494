#!/usr/bin/env python3
"""
spatial-pfn-treatment.py

generate a toy "causal mnist" dataset by overlaying a treatment marker (x)
and an outcome marker (o) onto mnist digits, then embed the resulting images
using dinov3 vit-l/16 cls embeddings. finally, run a quick pca sanity check
and save a few colored scatter plots.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# config
CANVAS_SIZE = 224
DIGIT_SIZE = 160
MARGIN = 12

RNG_SEED = 9

X_LEN, X_THICKNESS = 26, 5
O_RADIUS, O_THICKNESS = 18, 5

# treatment assignment: logit = alpha0 + alpha1*z + alpha2*x + alpha3*y
ALPHA0, ALPHA1, ALPHA2, ALPHA3 = -0.2, 1.0, 0.002, -0.002

# outcome: y = beta0 + beta1*t + beta2*z - beta3*dist(x,o) + noise
BETA0, BETA1, BETA2, BETA3 = 0.0, 1.5, 0.8, 0.01
NOISE_Y_STD = 0.2

COLOR_X, COLOR_O, COLOR_DIGIT = 245, 210, 255

TRAIN_N, TEST_N = 5000, 1000
OUT_DIR = Path("./mnist_causal")
DATA_ROOT = Path("./data")
ANALYSIS_DIR = OUT_DIR / "analysis"

REPO_DIR = Path(os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3"))
WEIGHTS = Path(
    os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def label_to_xy(label: int, canvas: int) -> Tuple[int, int]:
    # deterministic mapping from digit label to x marker position
    L = float(label)
    u = ((L * 0.20) + 0.30 * math.sin((L + 1) * 1.10)) % 1.0
    v = ((L * 0.24) + 0.30 * math.cos((L + 2) * 0.90)) % 1.0
    x = int(MARGIN + u * (canvas - 2 * MARGIN))
    y = int(MARGIN + v * (canvas - 2 * MARGIN))
    return x, y


def label_to_xy_alt(label: int, canvas: int) -> Tuple[int, int]:
    # second deterministic mapping for o marker
    L = float(label)
    u = ((L * 0.20) + 0.30 * math.cos((L + 3) * 0.70)) % 1.0
    v = ((L * 0.25) + 0.25 * math.sin((L + 4) * 1.30)) % 1.0
    x = int(MARGIN + u * (canvas - 2 * MARGIN))
    y = int(MARGIN + v * (canvas - 2 * MARGIN))
    return x, y


def compute_confounder(np_img28: np.ndarray) -> float:
    # hand-crafted scalar confounder in [0, 1]
    img = np_img28.astype(np.float32) / 255.0
    mean_int = float(img.mean())

    rows = np.arange(img.shape[0], dtype=np.float32)[:, None]
    mass = float(img.sum()) + 1e-6
    vcent = float((rows * img).sum() / mass) / (img.shape[0] - 1)

    z_raw = 0.6 * mean_int + 0.4 * (1.0 - vcent)
    return sigmoid(3.0 * (z_raw - 0.5))


def sample_treatment(z: float, xX: int, yX: int) -> Tuple[int, float]:
    logit = ALPHA0 + ALPHA1 * z + ALPHA2 * float(xX) + ALPHA3 * float(yX)
    p = sigmoid(logit)
    t = 1 if random.random() < p else 0
    return t, p


def sample_outcome(t: int, z: float, xX: int, yX: int, xO: int, yO: int) -> float:
    dist = math.hypot(float(xX - xO), float(yX - yO))
    noise = random.gauss(0.0, NOISE_Y_STD)
    return BETA0 + BETA1 * float(t) + BETA2 * z - BETA3 * dist + noise


def draw_centered_digit_on_canvas(img28: Image.Image) -> Image.Image:
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)

    digit = img28.resize((DIGIT_SIZE, DIGIT_SIZE), resample=Image.BICUBIC)
    digit = ImageOps.colorize(digit, black="black", white=(COLOR_DIGIT,) * 3).convert("L")

    x0 = (CANVAS_SIZE - DIGIT_SIZE) // 2
    y0 = (CANVAS_SIZE - DIGIT_SIZE) // 2
    canvas.paste(digit, (x0, y0))
    return canvas


def draw_X(draw: ImageDraw.ImageDraw, x: int, y: int, length: int, thickness: int, color: int) -> None:
    h = length // 2
    draw.line((x - h, y - h, x + h, y + h), fill=color, width=thickness)
    draw.line((x - h, y + h, x + h, y - h), fill=color, width=thickness)


def draw_O(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, thickness: int, color: int) -> None:
    bbox = (x - radius, y - radius, x + radius, y + radius)
    for k in range(thickness):
        draw.ellipse((bbox[0] - k, bbox[1] - k, bbox[2] + k, bbox[3] + k), outline=color)


def process_split(dataset, split_name: str, out_dir: Path, max_n: int, seed: int) -> Dict[str, int]:
    # generate one split and write images + metadata to disk
    set_all_seeds(seed)

    split_dir = out_dir / split_name
    img_dir = split_dir / "images"
    ensure_dir(img_dir)

    meta_path = split_dir / "metadata.csv"
    n = min(max_n, len(dataset))

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "split", "digit_label", "z_confounder", "x_X", "y_X", "x_O", "y_O", "T", "p_T", "Y"]
        )

        for idx in range(n):
            pil28, label = dataset[idx]
            np28 = np.array(pil28, dtype=np.uint8)

            z = compute_confounder(np28)
            xX, yX = label_to_xy(int(label), CANVAS_SIZE)
            xO, yO = label_to_xy_alt(int(label), CANVAS_SIZE)

            T, pT = sample_treatment(z, xX, yX)
            Y = sample_outcome(T, z, xX, yX, xO, yO)

            canvas = draw_centered_digit_on_canvas(pil28)
            draw = ImageDraw.Draw(canvas)

            if T == 1:
                draw_X(draw, xX, yX, X_LEN, X_THICKNESS, COLOR_X)
            else:
                draw_X(draw, xX, yX, X_LEN, max(1, X_THICKNESS // 2), 140)

            o_thick = max(2, int(O_THICKNESS + 2 * max(0.0, float(Y))))
            draw_O(draw, xO, yO, O_RADIUS, o_thick, COLOR_O)

            img_name = f"{split_name}_{idx:06d}.png"
            canvas.save(img_dir / img_name)

            writer.writerow(
                [
                    img_name,
                    split_name,
                    int(label),
                    round(z, 6),
                    xX,
                    yX,
                    xO,
                    yO,
                    int(T),
                    round(pT, 6),
                    round(float(Y), 6),
                ]
            )

    print(f"{split_name}: wrote {n} images + metadata", flush=True)
    return {"saved": n, "csv_rows": n}


def build_datasets(data_root: Path):
    train_ds = datasets.MNIST(root=str(data_root), train=True, download=True, transform=None)
    test_ds = datasets.MNIST(root=str(data_root), train=False, download=True, transform=None)
    return train_ds, test_ds


class CausalMNISTFolder(Dataset):
    def __init__(self, split_dir: Path):
        self.img_dir = split_dir / "images"
        self.meta = pd.read_csv(split_dir / "metadata.csv")
        self.meta.rename(columns=lambda c: c.strip(), inplace=True)
        self.files = self.meta["id"].tolist()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        img = Image.open(self.img_dir / fn).convert("L")
        return self.transform(img), self.meta.iloc[idx].to_dict()


def collate_fn(batch):
    imgs, metas = zip(*batch)
    return torch.stack(imgs, 0), list(metas)


def load_dino():
    print("loading dinov3 (local repo)...", flush=True)
    model = torch.hub.load(str(REPO_DIR), "dinov3_vitl16", source="local", pretrained=False)

    state = torch.load(str(WEIGHTS), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device: {device}", flush=True)
    return model, device


@torch.no_grad()
def extract_embeddings(model, device, loader):
    embeds: List[np.ndarray] = []
    metas: List[Dict] = []

    for imgs, rows in loader:
        imgs = imgs.to(device)
        out = model.get_intermediate_layers(imgs, n=1)[0]
        cls = out[:, 0].detach().cpu().numpy()
        embeds.append(cls)
        metas.extend(rows)

    return np.concatenate(embeds, axis=0), metas


def plot_pca(X2: np.ndarray, c: np.ndarray, title: str, outpath: Path, continuous: bool = False) -> None:
    plt.figure(figsize=(6, 5))

    if continuous:
        sc = plt.scatter(X2[:, 0], X2[:, 1], c=c, s=8, cmap="viridis", alpha=0.8)
        cb = plt.colorbar(sc)
        cb.set_label(title)
    else:
        unique_vals = np.unique(c)
        cmap = plt.cm.get_cmap("tab10", len(unique_vals))
        for i, val in enumerate(unique_vals):
            m = c == val
            plt.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.8, color=cmap(i), label=str(val))
        plt.legend(title=title, markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(title)
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {outpath}", flush=True)


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(DATA_ROOT)
    ensure_dir(ANALYSIS_DIR)

    print("generating overlays...", flush=True)
    train_ds, test_ds = build_datasets(DATA_ROOT)
    process_split(train_ds, "train", OUT_DIR, TRAIN_N, RNG_SEED)
    process_split(test_ds, "test", OUT_DIR, TEST_N, RNG_SEED + 1)

    model, device = load_dino()

    train_folder = CausalMNISTFolder(OUT_DIR / "train")
    test_folder = CausalMNISTFolder(OUT_DIR / "test")

    num_workers = 4
    if os.environ.get("SLURM_JOB_ID") is None and os.name == "nt":
        num_workers = 0

    train_loader = DataLoader(
        train_folder, batch_size=64, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_folder, batch_size=64, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    print("embedding...", flush=True)
    emb_tr, meta_tr = extract_embeddings(model, device, train_loader)
    emb_te, meta_te = extract_embeddings(model, device, test_loader)

    np.save(ANALYSIS_DIR / "emb_train.npy", emb_tr)
    np.save(ANALYSIS_DIR / "emb_test.npy", emb_te)
    pd.DataFrame(meta_tr).to_csv(ANALYSIS_DIR / "meta_train.csv", index=False)
    pd.DataFrame(meta_te).to_csv(ANALYSIS_DIR / "meta_test.csv", index=False)

    print("pca sanity check...", flush=True)
    E = np.vstack([emb_tr, emb_te])
    M = pd.concat([pd.DataFrame(meta_tr), pd.DataFrame(meta_te)], ignore_index=True)

    pca = PCA(n_components=2, random_state=RNG_SEED)
    X2 = pca.fit_transform(E)
    np.save(ANALYSIS_DIR / "pca2.npy", X2)

    plot_pca(X2, M["T"].astype(int).to_numpy(), "pca by treatment t", ANALYSIS_DIR / "pca_by_T.png")
    plot_pca(
        X2,
        M["Y"].astype(float).to_numpy(),
        "pca by outcome y",
        ANALYSIS_DIR / "pca_by_Y.png",
        continuous=True,
    )
    plot_pca(
        X2,
        M["z_confounder"].astype(float).to_numpy(),
        "pca by confounder z",
        ANALYSIS_DIR / "pca_by_z.png",
        continuous=True,
    )
    plot_pca(
        X2,
        M["digit_label"].astype(int).to_numpy(),
        "pca by digit",
        ANALYSIS_DIR / "pca_by_digit.png",
    )

    with open(ANALYSIS_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "n_train": len(train_folder),
                "n_test": len(test_folder),
                "embed_dim": int(E.shape[1]),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            f,
            indent=2,
        )

    print("done.", flush=True)


if __name__ == "__main__":
    main()
