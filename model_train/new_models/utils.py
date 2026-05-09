import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    max_labels: int = 50,
) -> None:
    cm = cm.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")

    n = cm.shape[0]
    if n <= max_labels:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i) for i in ticks], rotation=90)
        ax.set_yticklabels([str(i) for i in ticks])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@dataclass
class RunMetrics:
    task: str
    model: str
    dataset: str
    train_subsets: List[str]
    test_subset: str
    num_classes: int
    train_samples: int
    val_samples: int
    test_samples: int
    best_val_acc: float
    test_acc: float
    seed: int
    epochs: int
    batch_size: int
    lr: float
    extras: Dict[str, Any]


def save_metrics_json(metrics: RunMetrics, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    payload = asdict(metrics)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
    acc = float(correct / max(1, total))
    cm = confusion_matrix_counts(np.array(y_true), np.array(y_pred), num_classes=num_classes)
    return acc, cm


def plot_training_history(
    train_loss: List[float],
    val_acc: List[float],
    out_path: str,
    test_acc: Optional[float] = None,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_loss, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(val_acc, label="Val Acc")
    if test_acc is not None:
        ax2.axhline(float(test_acc), linestyle="--", linewidth=1.5, label=f"Test Acc={test_acc:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

