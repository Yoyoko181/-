import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_models.datasets import (
    FootstepSignalDataset,
    build_index_from_interim,
    stratified_split_indices,
)
from new_models.models_bilstm_attn import BiLSTMAttnClassifier
from new_models.models_hybrid import HybridResNetBiLSTMAttn, hybrid_variant
from new_models.models_resnet1d_se import ResNet1DSE
from new_models.utils import (
    AverageMeter,
    RunMetrics,
    ensure_dir,
    evaluate_classifier,
    get_device,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics_json,
    seed_everything,
)


def parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_closed_set_class_mapping(
    train_samples, train_persons, test_samples, test_persons
) -> Tuple[Dict[str, int], List[str]]:
    common = sorted(list(set(train_persons).intersection(set(test_persons))))
    def key(p: str) -> int:
        if p.startswith("P") and p[1:].isdigit():
            return int(p[1:])
        return 10**9
    common.sort(key=key)
    class_to_idx = {p: i for i, p in enumerate(common)}
    train_samples = [s for s in train_samples if s.person in class_to_idx]
    test_samples = [s for s in test_samples if s.person in class_to_idx]
    return class_to_idx, common


def make_model(args, num_classes: int) -> nn.Module:
    if args.model == "resnet1d_se":
        return ResNet1DSE(
            num_classes=num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
        )
    if args.model == "bilstm_attn":
        return BiLSTMAttnClassifier(
            num_classes=num_classes,
            d_model=args.d_model,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
        )
    if args.model == "hybrid":
        v = hybrid_variant(args.hybrid_variant)
        return HybridResNetBiLSTMAttn(
            num_classes=num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
            use_multiscale=v.use_multiscale,
            use_se=v.use_se,
            use_lstm=v.use_lstm,
            use_attn=v.use_attn,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            attn_heads=args.attn_heads,
        )
    raise ValueError(f"Unknown model={args.model}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    meter = AverageMeter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        meter.update(float(loss.item()), n=int(y.numel()))
    return meter.avg


class SyntheticSignalDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        xi = self.x[idx]
        if xi.ndim == 1:
            xi = xi.unsqueeze(0)
        yi = self.y[idx]
        return xi, yi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interim_root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="A3")
    p.add_argument("--train_subsets", type=str, default="A3_1,A3_3")
    p.add_argument("--test_subset", type=str, default="A3_2")
    p.add_argument("--test_subsets", type=str, default="")
    p.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "rms", "none"])
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--model",
        type=str,
        default="resnet1d_se",
        choices=["resnet1d_se", "bilstm_attn", "hybrid"],
    )
    p.add_argument(
        "--hybrid_variant",
        type=str,
        default="full",
        choices=["base", "ms", "ms_se", "ms_se_lstm", "full"],
    )
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--results_dir", type=str, default="")
    p.add_argument("--synthetic_pt", type=str, default="")
    p.add_argument("--synthetic_ratio", type=float, default=0.0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_subsets = parse_list(args.train_subsets)
    if not train_subsets:
        raise ValueError("train_subsets is empty")

    if args.test_subsets.strip():
        test_subsets = parse_list(args.test_subsets)
    else:
        test_subsets = [args.test_subset]

    train_samples, train_persons = build_index_from_interim(
        args.interim_root, args.dataset, train_subsets
    )
    test_samples, test_persons = build_index_from_interim(
        args.interim_root, args.dataset, test_subsets
    )

    class_to_idx, persons = build_closed_set_class_mapping(
        train_samples, train_persons, test_samples, test_persons
    )
    num_classes = len(persons)
    if num_classes < 2:
        raise RuntimeError("Closed-set common persons < 2. Check interim_root/dataset/subsets.")

    train_samples = [s for s in train_samples if s.person in class_to_idx]
    test_samples = [s for s in test_samples if s.person in class_to_idx]

    y_all = np.array([class_to_idx[s.person] for s in train_samples], dtype=np.int64)
    train_idx, val_idx = stratified_split_indices(y_all, val_ratio=args.val_ratio, seed=args.seed)
    train_split = [train_samples[i] for i in train_idx.tolist()]
    val_split = [train_samples[i] for i in val_idx.tolist()]

    subset_to_idx = {s: i for i, s in enumerate(sorted(set(train_subsets + test_subsets)))}

    ds_train = FootstepSignalDataset(
        train_split, class_to_idx=class_to_idx, subset_to_idx=subset_to_idx, normalize=args.normalize
    )
    ds_val = FootstepSignalDataset(
        val_split, class_to_idx=class_to_idx, subset_to_idx=subset_to_idx, normalize=args.normalize
    )
    ds_test = FootstepSignalDataset(
        test_samples, class_to_idx=class_to_idx, subset_to_idx=subset_to_idx, normalize=args.normalize
    )

    extras_train_ds = ds_train
    if args.synthetic_pt.strip() and args.synthetic_ratio > 0:
        syn = torch.load(args.synthetic_pt, map_location="cpu")
        syn_x = syn["x"]
        syn_y = syn["y"]
        syn_persons = syn.get("persons", None)
        if syn_persons is None:
            raise ValueError("synthetic_pt missing persons list")
        syn_persons = list(syn_persons)
        keep_idx = []
        mapped_y = []
        for i in range(int(syn_y.numel())):
            cls = int(syn_y[i].item())
            if cls < 0 or cls >= len(syn_persons):
                continue
            person = syn_persons[cls]
            if person not in class_to_idx:
                continue
            keep_idx.append(i)
            mapped_y.append(class_to_idx[person])
        if keep_idx:
            keep_idx_t = torch.tensor(keep_idx, dtype=torch.long)
            syn_x = syn_x[keep_idx_t]
            syn_y_m = torch.tensor(mapped_y, dtype=torch.long)
            max_n = int(len(ds_train) * float(args.synthetic_ratio))
            if max_n > 0 and syn_x.size(0) > max_n:
                g = torch.Generator().manual_seed(args.seed)
                perm = torch.randperm(syn_x.size(0), generator=g)[:max_n]
                syn_x = syn_x[perm]
                syn_y_m = syn_y_m[perm]
            syn_ds = SyntheticSignalDataset(syn_x.float(), syn_y_m)
            extras_train_ds = ConcatDataset([ds_train, syn_ds])
            print(f"Using synthetic: {args.synthetic_pt} -> {len(syn_ds)} samples")

    train_loader = DataLoader(
        extras_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.results_dir.strip():
        out_dir = args.results_dir
    else:
        model_tag = args.model
        if args.model == "hybrid":
            model_tag = f"{args.model}_{args.hybrid_variant}"
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "new_models",
            f"{model_tag}_{args.dataset}_{'-'.join(train_subsets)}__to__{'-'.join(test_subsets)}",
        )
    out_dir = os.path.abspath(out_dir)
    ensure_dir(out_dir)
    print(f"Results: {out_dir}")

    model = make_model(args, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = os.path.join(out_dir, "best_model.pth")
    train_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss_hist.append(loss)
        val_acc, _ = evaluate_classifier(model, val_loader, device=device, num_classes=num_classes)
        val_acc_hist.append(val_acc)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, cm = evaluate_classifier(model, test_loader, device=device, num_classes=num_classes)
    print(f"Best Val Acc: {best_val:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    metrics = RunMetrics(
        task="closed_set_person_id",
        model=f"{args.model}:{args.hybrid_variant}" if args.model == "hybrid" else args.model,
        dataset=args.dataset,
        train_subsets=train_subsets,
        test_subset=",".join(test_subsets),
        num_classes=num_classes,
        train_samples=len(extras_train_ds),
        val_samples=len(ds_val),
        test_samples=len(ds_test),
        best_val_acc=float(best_val),
        test_acc=float(test_acc),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        extras={
            "normalize": args.normalize,
            "dropout": args.dropout,
            "base_channels": args.base_channels,
            "d_model": args.d_model,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "attn_heads": args.attn_heads,
            "hybrid_variant": args.hybrid_variant,
            "subset_to_idx": subset_to_idx,
            "synthetic_pt": args.synthetic_pt,
            "synthetic_ratio": args.synthetic_ratio,
        },
    )
    save_metrics_json(metrics, os.path.join(out_dir, "metrics.json"))
    plot_training_history(
        train_loss_hist,
        val_acc_hist,
        os.path.join(out_dir, "training_history.png"),
        test_acc=test_acc,
    )
    plot_confusion_matrix(
        cm,
        os.path.join(out_dir, "confusion_matrix_test.png"),
        normalize=True,
        title="Test Confusion Matrix (row-normalized)",
    )

    print(f"Saved: {best_path}")
    print(f"Saved: {os.path.join(out_dir, 'metrics.json')}")
    print(f"Saved: {os.path.join(out_dir, 'confusion_matrix_test.png')}")


if __name__ == "__main__":
    main()
