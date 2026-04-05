import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import CatsDogsDataset, build_samples, get_transforms, split_samples
from src.model import build_model
from src.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    image_size = data_cfg["image_size"]

    # Copy dataset to work_dir for faster I/O on Colab
    work_dir = data_cfg.get("work_dir")
    if work_dir and work_dir != raw_dir:
        if not os.path.exists(work_dir):
            print(f"Copying data to {work_dir} ...")
            import shutil
            shutil.copytree(raw_dir, work_dir)
        raw_dir = work_dir

    samples = build_samples(raw_dir)
    train_samples, val_samples = split_samples(
        samples, val_split=data_cfg["val_split"], seed=seed
    )
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = CatsDogsDataset(train_samples, transform=get_transforms("train", image_size))
    val_dataset   = CatsDogsDataset(val_samples,   transform=get_transforms("val",   image_size))

    num_workers = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model_cfg = config["model"]
    model = build_model(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
    )

    if config["training"].get("resume"):
        from src.model import load_checkpoint
        ckpt_path = os.path.join(config["output"]["dir"], "best.pth")
        model, start_epoch, best_metric = load_checkpoint(model, ckpt_path)
        print(f"Resumed from {ckpt_path} (epoch={start_epoch}, best_acc={best_metric:.4f})")

    # --- Train ---
    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(config, model, device, output_dir)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
