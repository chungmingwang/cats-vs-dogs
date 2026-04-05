import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CatsDogsDataset, build_samples, get_transforms, split_samples
from src.model import build_model, load_checkpoint


def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # P(dog)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_score.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_score)


def plot_confusion_matrix(y_true, y_pred, output_path, class_names=("cat", "dog")):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true, y_score, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")
    return auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cats vs Dogs classifier")
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pth). Defaults to <output_dir>/best.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    raw_dir = data_cfg["raw_dir"]

    samples = build_samples(raw_dir)
    _, val_samples = split_samples(
        samples,
        val_split=data_cfg["val_split"],
        seed=config["training"]["seed"],
    )
    print(f"Evaluation set: {len(val_samples)} images")

    val_dataset = CatsDogsDataset(val_samples, transform=get_transforms("val", image_size))
    val_loader  = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
    model_cfg = config["model"]
    model = build_model(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
    )

    ckpt_path = args.checkpoint or os.path.join(config["output"]["dir"], "best.pth")
    model, epoch, best_metric = load_checkpoint(model, ckpt_path)
    model = model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}  (saved at epoch {epoch})")

    # --- Inference ---
    y_true, y_pred, y_score = collect_predictions(model, val_loader, device)

    # --- Metrics ---
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)

    print("\n========== Evaluation Results ==========")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    # --- Plots ---
    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, "confusion_matrix.png"))
    auc = plot_roc_curve(y_true, y_score, os.path.join(output_dir, "roc_curve.png"))
    print(f"  AUC-ROC   : {auc:.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()
