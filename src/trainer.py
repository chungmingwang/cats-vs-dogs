import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


class Trainer:
    def __init__(self, config: dict, model: nn.Module, device: torch.device, output_dir: str):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg.get("early_stopping_patience", 5)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=0.0001,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )
            val_loss, val_acc = validate(
                self.model, val_loader, self.criterion, self.device
            )
            self.scheduler.step()

            print(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "best_metric": best_acc,
                    },
                    path=f"{self.output_dir}/best.pth",
                )
                print(f"  -> Saved best checkpoint (val_acc={best_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\nTraining done. Best val_acc={best_acc:.4f}")
