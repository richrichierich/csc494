#!/usr/bin/env python3
"""
synthpfn-rollout.py

rollout viz + per-digit pca plots
"""

import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

# paths
REPO_DIR   = os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3")
WEIGHTS    = os.path.expanduser("~/projects/aip-rahulgk/richguo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
OUT_DIR    = Path("./mnist_causal")
ANALYSIS_DIR = OUT_DIR/"analysis"
ROLLOUT_DIR  = OUT_DIR/"rollout_plots"
CLASS_PLOTS_DIR = OUT_DIR/"class_pca_plots"

os.makedirs(ROLLOUT_DIR, exist_ok=True)
os.makedirs(CLASS_PLOTS_DIR, exist_ok=True)

print("py", os.popen("python --version").read().strip(), flush=True)

# load dino
print("load dino", flush=True)
dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', pretrained=False)
state_dict = torch.load(WEIGHTS, map_location="cpu")
dinov3.load_state_dict(state_dict, strict=False)
dinov3.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov3.to(device)
print(f"device={device}", flush=True)

# preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def pil_to_tensor(pil_img):
    # to tensor
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return transform(pil_img).unsqueeze(0).to(device)

# rollout
@torch.no_grad()
def attention_rollout(model, x):
    # attn rollout
    attns = []
    handles = []

    def hook(module, input, output):
        attns.append(output[1].detach() if isinstance(output, tuple) else output.detach())

    for blk in model.blocks:
        h = blk.attn.register_forward_hook(hook)
        handles.append(h)

    _ = model(x)

    for h in handles:
        h.remove()

    rollout = None
    for attn in attns:
        attn = attn.mean(1)
        attn = attn[0]
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = attn + torch.eye(attn.size(0), device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else attn @ rollout

    mask = rollout[0, 1:]
    num_tokens = mask.shape[0]
    grid = int(np.ceil(np.sqrt(num_tokens)))
    usable = grid * grid

    # pad/trim
    if num_tokens < usable:
        pad = torch.zeros(usable - num_tokens, device=mask.device)
        mask = torch.cat([mask, pad])
    elif num_tokens > usable:
        mask = mask[:usable]

    mask = mask.reshape(grid, grid).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

def visualize_rollout(model, pil_img, out_path):
    # save viz
    tensor = pil_to_tensor(pil_img)
    mask = attention_rollout(model, tensor)

    mask_resized = cv2.resize(mask, pil_img.size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.array(pil_img).astype(np.float32)
    if overlay.ndim == 2:
        overlay = np.repeat(overlay[..., None], 3, axis=2)

    overlay = 0.6 * overlay + 0.4 * heatmap
    overlay = np.uint8(np.clip(overlay, 0, 255))

    Image.fromarray(overlay).save(out_path)
    print("saved", out_path, flush=True)

# one per class
train_imgs_dir = OUT_DIR / "train" / "images"
train_meta = pd.read_csv(OUT_DIR / "train" / "metadata.csv")

print("rollout per digit", flush=True)
for digit in range(10):
    row = train_meta[train_meta["digit_label"] == digit].iloc[0]
    img_path = train_imgs_dir / row["id"]
    pil_img = Image.open(img_path).convert("L")
    out_path = ROLLOUT_DIR / f"rollout_digit_{digit}.png"
    visualize_rollout(dinov3, pil_img, out_path)

# load embeds
print("load embeds", flush=True)
E = np.load(ANALYSIS_DIR / "emb_train.npy")
M = pd.read_csv(ANALYSIS_DIR / "meta_train.csv")
print("cols", M.columns.tolist(), flush=True)

# pca
pca = PCA(n_components=2, random_state=9)
X2 = pca.fit_transform(E)
np.save(CLASS_PLOTS_DIR / "pca2.npy", X2)

def plot_pca(X2, c, title, outpath, continuous=False):
    # plot helper
    plt.figure(figsize=(6,5))
    if continuous:
        sc = plt.scatter(X2[:,0], X2[:,1], c=c, s=8, cmap="viridis", alpha=0.8)
        cbar = plt.colorbar(sc)
        cbar.set_label(title)
    else:
        unique_vals = np.unique(c)
        cmap = plt.cm.get_cmap("tab10", len(unique_vals))
        for i, val in enumerate(unique_vals):
            mask = (c == val)
            plt.scatter(X2[mask,0], X2[mask,1], s=8, alpha=0.8, color=cmap(i), label=str(val))
        plt.legend(title=title, markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath,dpi=200,bbox_inches="tight")
    plt.close()
    print("saved", outpath, flush=True)

# per digit plots
print("class pca", flush=True)
for digit in range(10):
    mask = M["digit_label"] == digit
    Xd = X2[mask]
    Md = M[mask]
    print(f"digit {digit} n={Xd.shape[0]}", flush=True)

    plot_pca(
        Xd,
        Md["T"].astype(int).to_numpy(),
        f"Digit {digit} - Treatment T",
        CLASS_PLOTS_DIR / f"pca_digit{digit}_T.png",
    )
    plot_pca(
        Xd,
        Md["z_confounder"].astype(float).to_numpy(),
        f"Digit {digit} - Confounder z",
        CLASS_PLOTS_DIR / f"pca_digit{digit}_z.png",
        continuous=True,
    )
    plot_pca(
        Xd,
        Md["Y"].astype(float).to_numpy(),
        f"Digit {digit} - Outcome Y",
        CLASS_PLOTS_DIR / f"pca_digit{digit}_Y.png",
        continuous=True,
    )

print("done", flush=True)
print("rollouts", ROLLOUT_DIR, flush=True)
print("pca", CLASS_PLOTS_DIR, flush=True)
