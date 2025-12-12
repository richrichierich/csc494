import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import os

#save
save_dir = '/home/richguo/projects/def-rahulgk/richguo/mnist_viz'
os.makedirs(save_dir, exist_ok=True)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tabicl import TabICLClassifier
import torch
torch.set_num_threads(8)


print("Loading MNIST...", flush=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#flatten + normalize
X_train_full = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
X_test_full  = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0


n_test = 200
X_test_small, _, y_test_small, _ = train_test_split(
    X_test_full, y_test,
    train_size=n_test,
    stratify=y_test,
    random_state=9
)

#params
train_sizes = [200, 500, 1000, 2000, 5000]        # vary training sizes
pca_components_list = [10, 25, 50, 100, 200]               # vary PCA components

for pca_features in pca_components_list:
    print(f"\n=== Running experiments with PCA={pca_features} components ===")
    accuracies = []  # reset per PCA config
    for n_train in train_sizes:
        print(f"\nTraining with {n_train} samples (PCA={pca_features})...")
        X_train_small, _, y_train_small, _ = train_test_split(
            X_train_full, y_train,
            train_size=n_train,
            stratify=y_train,
            random_state=9
        )

        # PCA reduction
        pca = PCA(n_components=pca_features)
        X_train_reduced = pca.fit_transform(X_train_small)
        X_test_reduced  = pca.transform(X_test_small)

        # Fit TabICLClassifier
        model = TabICLClassifier(device='cpu', verbose=False)
        model.fit(X_train_reduced, y_train_small)
        y_pred = model.predict(X_test_reduced)

        acc = accuracy_score(y_test_small, y_pred)
        print(f"Accuracy: {acc:.3f}")
        accuracies.append((n_train, acc))

    #plots
    plt.figure()
    plt.plot([n for n, _ in accuracies],
             [a for _, a in accuracies],
             marker='o')
    plt.title(f"Accuracy vs Training Set Size (PCA={pca_features})")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1)
    plt.grid(True)
    out_path = os.path.join(save_dir, f"accuracy_vs_train_pca{pca_features}.png")
    plt.savefig(out_path)
    plt.close()

print("viz in ", save_dir)
