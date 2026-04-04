from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model


def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str) -> tuple[nn.Module, int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    return model, epoch, best_metric
