#!e:\毕设\Terra-main\model_train\.venv\Scripts\python.exe
import argparse
from collections import Counter
import re
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FootstepImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _list_class_names(root: Path) -> list[str]:
    if not root.exists():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(names)


def _iter_images(cls_dir: Path) -> list[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]
    out: list[Path] = []
    for pat in exts:
        out.extend(sorted(cls_dir.glob(pat)))
    return out


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False).reshape(-1)
    x = x - float(np.mean(x))
    x = x / float(np.std(x) + 1e-8)
    return x


def _cwt_rgb(signal_1d: np.ndarray, scales: np.ndarray, wavelet: str, out_size: int) -> Image.Image:
    coeffs, _ = pywt.cwt(signal_1d, scales, wavelet)
    img = np.abs(coeffs).astype(np.float32, copy=False)
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    img = (img - vmin) / float((vmax - vmin) + 1e-8)
    img = np.clip(img, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps.get_cmap("jet")
    except Exception:
        cmap = cm.get_cmap("jet")
    rgb = cmap(img)[..., :3]
    rgb8 = (rgb * 255.0).astype(np.uint8)
    pil = Image.fromarray(rgb8, mode="RGB").resize((out_size, out_size), resample=Image.BILINEAR)
    return pil


def _load_signal_1d_from_mat(mat_path: Path) -> np.ndarray:
    d = scipy.io.loadmat(str(mat_path))
    keys = [k for k in d.keys() if not k.startswith("__")]
    for k in ["geo_data", "data", "signal", "x", "X"]:
        if k in d:
            arr = d[k]
            if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
                return np.asarray(arr, dtype=np.float32).reshape(-1)
    for k in keys:
        arr = d[k]
        if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
            x = np.asarray(arr, dtype=np.float32).reshape(-1)
            if x.ndim == 1 and x.size > 0:
                return x
    raise ValueError(f"{mat_path} 未找到可解析的一维数值信号，keys={keys}")


def _segments_random(signal: np.ndarray, segment_len: int, segments_per_file: int, seed: int) -> list[np.ndarray]:
    if signal.size < segment_len:
        return []
    x = _normalize_1d(signal)
    max_start = x.size - segment_len
    if max_start <= 0:
        return [x[:segment_len]]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=segments_per_file, endpoint=False)
    out: list[np.ndarray] = []
    for s in starts.tolist():
        seg = x[s : s + segment_len]
        if seg.size != segment_len:
            continue
        if np.allclose(seg, seg[0]):
            continue
        out.append(seg)
    return out


def _segments_energy_peaks(
    signal: np.ndarray,
    segment_len: int,
    segments_per_file: int,
    seed: int,
    energy_window: int,
    peak_k: float,
    min_peak_distance: int,
) -> list[np.ndarray]:
    if signal.size < segment_len:
        return []
    x = _normalize_1d(signal)
    w = int(max(8, energy_window))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    energy = np.convolve(x * x, kernel, mode="same")
    thr = float(np.mean(energy) + peak_k * (np.std(energy) + 1e-8))
    candidates = np.where((energy[1:-1] > energy[:-2]) & (energy[1:-1] >= energy[2:]) & (energy[1:-1] > thr))[0] + 1
    if candidates.size == 0:
        return []
    min_dist = int(max(1, min_peak_distance))
    kept: list[int] = []
    last = -10**18
    for idx in candidates.tolist():
        if idx - last >= min_dist:
            kept.append(idx)
            last = idx
    if not kept:
        return []
    rng = np.random.default_rng(seed)
    if len(kept) > segments_per_file:
        kept = rng.choice(np.array(kept, dtype=np.int64), size=segments_per_file, replace=False).tolist()
    max_start = x.size - segment_len
    half = segment_len // 2
    segments: list[np.ndarray] = []
    for peak in kept:
        start = int(peak - half)
        start = max(0, min(max_start, start))
        seg = x[start : start + segment_len]
        if seg.size != segment_len:
            continue
        if np.allclose(seg, seg[0]):
            continue
        segments.append(seg)
    return segments


def _iter_people_raw(subset_dir: Path) -> list[tuple[str, list[Path]]]:
    items: list[tuple[str, list[Path]]] = []
    for person_dir in subset_dir.iterdir():
        if not person_dir.is_dir():
            continue
        if not re.fullmatch(r"P\d+", person_dir.name):
            continue
        mats = sorted(person_dir.glob("*.mat"))
        if mats:
            items.append((person_dir.name, mats))
    items.sort(key=lambda x: int(x[0][1:]))
    return items


def _count_images_in_root(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        total += len(_iter_images(p))
    return total


def _ensure_cwt_images_from_raw(
    foot_db_root: Path,
    dataset: str,
    subset: str,
    out_root: Path,
    *,
    image_size: int,
    image_ext: str,
    wavelet: str,
    scales_max: int,
    segment_len: int,
    segments_per_file: int,
    max_files_per_person: int,
    max_persons: int,
    mode: str,
    energy_window: int,
    peak_k: float,
    min_peak_distance: int,
) -> Path:
    subset_out = out_root / dataset / subset
    subset_out.mkdir(parents=True, exist_ok=True)
    existing = _count_images_in_root(subset_out)

    subset_in = foot_db_root / dataset / subset
    if not subset_in.exists():
        raise FileNotFoundError(f"找不到 raw 数据目录: {subset_in}")

    people = _iter_people_raw(subset_in)
    if max_persons and max_persons > 0:
        people = people[:max_persons]
    if not people:
        raise RuntimeError(f"{subset_in} 下未找到任何 Pxx/*.mat")

    scales = np.arange(1, int(scales_max) + 1)
    files_done = 0
    images_done = 0
    existing_people = 0
    for person_name, _ in people:
        if (subset_out / person_name).exists() and _iter_images(subset_out / person_name):
            existing_people += 1
    if existing > 0:
        print(
            f"CWT 已存在: {subset_out} -> images={existing} people_with_images={existing_people}/{len(people)}",
            flush=True,
        )
    print(f"检查缺失并补齐 CWT: {subset} -> {subset_out}", flush=True)
    for person_name, mats in people:
        person_dir = subset_out / person_name
        if person_dir.exists() and _iter_images(person_dir):
            continue
        mats = mats[:max_files_per_person] if max_files_per_person and max_files_per_person > 0 else mats
        for mat_path in mats:
            files_done += 1
            seed = zlib.adler32(str(mat_path).encode("utf-8")) % 1_000_000
            signal = _load_signal_1d_from_mat(mat_path)
            if mode == "energy_peak":
                segs = _segments_energy_peaks(
                    signal,
                    segment_len=segment_len,
                    segments_per_file=segments_per_file,
                    seed=seed,
                    energy_window=energy_window,
                    peak_k=peak_k,
                    min_peak_distance=min_peak_distance,
                )
                if not segs:
                    segs = _segments_random(signal, segment_len, segments_per_file, seed)
            else:
                segs = _segments_random(signal, segment_len, segments_per_file, seed)

            for i, seg in enumerate(segs):
                img = _cwt_rgb(seg, scales=scales, wavelet=wavelet, out_size=image_size)
                out_path = subset_out / person_name / f"{mat_path.stem}__seg{i}.{image_ext}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    continue
                img.save(str(out_path))
                images_done += 1

            if files_done % 20 == 0:
                print(f"生成进度 {subset}: files={files_done} images={images_done}", flush=True)

    print(f"完成补齐 {subset}: files_processed={files_done} images_written={images_done}", flush=True)
    return subset_out


def _collect_image_paths_from_roots(
    roots: list[Path],
    class_to_idx: dict[str, int],
    class_names: list[str],
    limit_per_class: int,
) -> tuple[list[str], list[int]]:
    all_image_paths: list[str] = []
    all_labels: list[int] = []
    for root in roots:
        for cls_name in class_names:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            img_paths = _iter_images(cls_dir)
            if limit_per_class and limit_per_class > 0:
                img_paths = img_paths[:limit_per_class]
            if not img_paths:
                continue
            label = class_to_idx[cls_name]
            all_image_paths.extend([str(p) for p in img_paths])
            all_labels.extend([label] * len(img_paths))
    return all_image_paths, all_labels


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total else 0.0


def _predict(model: nn.Module, loader: DataLoader) -> tuple[list[int], list[int], float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(predicted.detach().cpu().numpy().astype(int).tolist())
    acc = correct / total if total else 0.0
    return y_true, y_pred, float(acc)


def _save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    *,
    num_classes: int,
    class_names: list[str],
    out_path: Path,
) -> None:
    cmx = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    denom = cmx.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    cmn = cmx.astype(np.float32) / denom.astype(np.float32)
    plt.figure(figsize=(10, 8))
    plt.imshow(cmn, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    if num_classes <= 30:
        ticks = np.arange(num_classes)
        plt.xticks(ticks, class_names, rotation=90, fontsize=6)
        plt.yticks(ticks, class_names, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_per_class", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--foot_db_root", type=str, default=r"e:\毕设\Foot_database")
    parser.add_argument("--dataset", type=str, default="A3")
    parser.add_argument("--train_subsets", type=str, default="A3_1,A3_3")
    parser.add_argument("--test_subset", type=str, default="A3_2")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--cwt_out_root", type=str, default=str((Path(__file__).resolve().parent / "cwt_images").resolve()))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_ext", type=str, default="jpg")
    parser.add_argument("--wavelet", type=str, default="morl")
    parser.add_argument("--scales_max", type=int, default=256)
    parser.add_argument("--segment_len", type=int, default=1500)
    parser.add_argument("--segments_per_file", type=int, default=10)
    parser.add_argument("--max_files_per_person", type=int, default=0)
    parser.add_argument("--max_persons", type=int, default=0)
    parser.add_argument("--mode", type=str, default="energy_peak", choices=["energy_peak", "random"])
    parser.add_argument("--energy_window", type=int, default=256)
    parser.add_argument("--peak_k", type=float, default=2.5)
    parser.add_argument("--min_peak_distance", type=int, default=1024)
    args = parser.parse_args()

    _set_seed(args.seed)

    foot_db_root = Path(args.foot_db_root)
    cwt_out_root = Path(args.cwt_out_root)
    train_subsets = [s.strip() for s in args.train_subsets.split(",") if s.strip()]
    test_subset = args.test_subset.strip()
    if not train_subsets:
        raise ValueError("train_subsets 不能为空")
    if not test_subset:
        raise ValueError("test_subset 不能为空")

    if not args.skip_generate:
        for subset in sorted(set(train_subsets + [test_subset])):
            _ensure_cwt_images_from_raw(
                foot_db_root=foot_db_root,
                dataset=args.dataset,
                subset=subset,
                out_root=cwt_out_root,
                image_size=args.image_size,
                image_ext=args.image_ext,
                wavelet=args.wavelet,
                scales_max=args.scales_max,
                segment_len=args.segment_len,
                segments_per_file=args.segments_per_file,
                max_files_per_person=args.max_files_per_person,
                max_persons=args.max_persons,
                mode=args.mode,
                energy_window=args.energy_window,
                peak_k=args.peak_k,
                min_peak_distance=args.min_peak_distance,
            )

    train_roots = [cwt_out_root / args.dataset / subset for subset in train_subsets]
    test_root = cwt_out_root / args.dataset / test_subset
    for p in train_roots + [test_root]:
        if not p.exists():
            raise FileNotFoundError(f"找不到图片目录: {p}")

    train_class_sets = [set(_list_class_names(p)) for p in train_roots]
    train_classes = set.union(*train_class_sets) if train_class_sets else set()
    test_classes = set(_list_class_names(test_root))
    common_classes = sorted(train_classes & test_classes)
    if not common_classes:
        common_classes = sorted(train_classes | test_classes)

    class_to_idx = {name: i for i, name in enumerate(common_classes)}
    num_classes = len(common_classes)
    if num_classes == 0:
        raise RuntimeError("未在图片目录中找到任何类别子文件夹")

    train_paths, train_labels = _collect_image_paths_from_roots(
        train_roots,
        class_to_idx=class_to_idx,
        class_names=common_classes,
        limit_per_class=args.limit_per_class,
    )
    test_paths, test_labels = _collect_image_paths_from_roots(
        [test_root],
        class_to_idx=class_to_idx,
        class_names=common_classes,
        limit_per_class=args.limit_per_class,
    )

    if not train_paths:
        raise RuntimeError(f"训练集图片为空: {train_roots}")
    if not test_paths:
        raise RuntimeError(f"测试集图片为空: {test_root}")

    print(f"Device: {DEVICE}", flush=True)
    print(f"Train roots: {[str(p) for p in train_roots]} -> images={len(train_paths)}", flush=True)
    print(f"Test root: {str(test_root)} -> images={len(test_paths)}", flush=True)
    print(f"Num classes: {num_classes}", flush=True)

    if len(train_labels) < 2:
        val_paths = list(train_paths)
        val_labels = list(train_labels)
    else:
        val_count = max(1, int(len(train_labels) * 0.2))
        val_count = min(val_count, len(train_labels) - 1)
        stratify = train_labels
        counts = Counter(train_labels)
        if not counts or min(counts.values()) < 2:
            stratify = None
        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths,
                train_labels,
                test_size=val_count,
                random_state=args.seed,
                stratify=stratify,
            )
        except ValueError:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths,
                train_labels,
                test_size=val_count,
                random_state=args.seed,
                stratify=None,
            )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = FootstepImageDataset(train_paths, train_labels, transform=transform)
    val_dataset = FootstepImageDataset(val_paths, val_labels, transform=transform)
    test_dataset = FootstepImageDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    weights = ResNet18_Weights.DEFAULT if args.use_pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results_dir = Path(__file__).resolve().parents[1] / "results" / "material"
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "best_footstep_model.pth"

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    print(f"Start training for {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        val_acc = _evaluate(model, val_loader)
        history["train_loss"].append(float(epoch_loss))
        history["val_acc"].append(float(val_acc))
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_model_path))

    if best_model_path.exists():
        try:
            state_dict = torch.load(str(best_model_path), map_location=DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(str(best_model_path), map_location=DEVICE)
        model.load_state_dict(state_dict)

    y_true_test, y_pred_test, test_acc = _predict(model, test_loader)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Acc")
    plt.axhline(test_acc, linestyle="--", linewidth=1.5, label=f"Test Acc: {test_acc:.4f}")
    plt.title("Validation Accuracy")
    plt.legend()
    history_path = results_dir / "training_history.png"
    plt.savefig(str(history_path))
    plt.close()

    cm_path = results_dir / "confusion_matrix_test.png"
    _save_confusion_matrix(
        y_true_test,
        y_pred_test,
        num_classes=num_classes,
        class_names=common_classes,
        out_path=cm_path,
    )

    metrics = {
        "task": "cross_material",
        "device": str(DEVICE),
        "dataset": args.dataset,
        "train_subsets": train_subsets,
        "test_subset": test_subset,
        "train_roots": [str(p) for p in train_roots],
        "test_root": str(test_root),
        "num_classes": int(num_classes),
        "num_train_images": int(len(train_dataset)),
        "num_val_images": int(len(val_dataset)),
        "num_test_images": int(len(test_dataset)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "use_pretrained": bool(args.use_pretrained),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "best_model_path": str(best_model_path),
        "training_history_path": str(history_path),
        "confusion_matrix_test_path": str(cm_path),
    }
    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Best Val Acc: {best_val_acc:.4f}", flush=True)
    print(f"Test Acc (Cross-Material): {test_acc:.4f}", flush=True)
    print(f"Saved: {best_model_path}", flush=True)
    print(f"Saved: {history_path}", flush=True)
    print(f"Saved: {metrics_path}", flush=True)
    print(f"Saved: {cm_path}", flush=True)


if __name__ == "__main__":
    main()
