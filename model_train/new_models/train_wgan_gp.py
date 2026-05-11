import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_models.datasets import build_index_from_interim
from new_models.utils import AverageMeter, ensure_dir, get_device, seed_everything
from new_models.wgan_gp import ConditionalMLPCritic, ConditionalMLPGenerator, gradient_penalty


class TensorDataset1D(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_subset_tensor(
    interim_root: str,
    dataset: str,
    subset: str,
    normalize: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, int]]:
    samples, persons = build_index_from_interim(interim_root, dataset, [subset])
    persons = sorted(persons, key=lambda p: int(p[1:]) if p.startswith("P") and p[1:].isdigit() else 10**9)
    class_to_idx = {p: i for i, p in enumerate(persons)}

    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    xs: List[np.ndarray] = []
    ys: List[int] = []

    cache: Dict[str, np.ndarray] = {}
    for s in samples:
        if s.person not in class_to_idx:
            continue
        if s.mat_path not in cache:
            import scipy.io as sio

            d = sio.loadmat(s.mat_path)
            cache[s.mat_path] = np.asarray(d["footstep_feat"], dtype=np.float32)
        sig = cache[s.mat_path][s.row].astype(np.float32)
        if normalize == "zscore":
            m = float(sig.mean())
            sd = float(sig.std())
            if sd < 1e-6:
                sd = 1.0
            sig = (sig - m) / sd
        elif normalize == "rms":
            rms = float(np.sqrt(np.mean(sig * sig)))
            if rms < 1e-6:
                rms = 1.0
            sig = sig / rms
        xs.append(sig)
        ys.append(class_to_idx[s.person])

    x = torch.from_numpy(np.stack(xs, axis=0))
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, persons, class_to_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interim_root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="A3")
    p.add_argument("--subset", type=str, default="A3_1")
    p.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "rms", "none"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    x, y, persons, class_to_idx = load_subset_tensor(
        args.interim_root, args.dataset, args.subset, args.normalize, args.seed
    )
    num_classes = len(persons)
    if num_classes < 2:
        raise RuntimeError("num_classes < 2")
    if int(x.size(0)) < 2:
        raise RuntimeError("num_samples < 2")

    ds = TensorDataset1D(x, y)
    bs = int(min(args.batch_size, int(x.size(0))))
    if bs < 2:
        raise RuntimeError("batch_size < 2 after adjustment")
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    if args.out_dir.strip():
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "results",
                "new_models_wgan",
                f"wgan_gp_{args.dataset}_{args.subset}",
            )
        )
    ensure_dir(out_dir)
    print(f"Out: {out_dir}")

    gen = ConditionalMLPGenerator(
        num_classes=num_classes,
        z_dim=args.z_dim,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        out_len=x.size(1),
    ).to(device)
    critic = ConditionalMLPCritic(
        num_classes=num_classes,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        in_len=x.size(1),
    ).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0.0, 0.9))

    step = 0
    for epoch in range(1, args.epochs + 1):
        d_meter = AverageMeter()
        g_meter = AverageMeter()
        for real, yb in loader:
            real = real.to(device)
            yb = yb.to(device)
            b = real.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(b, args.z_dim, device=device)
                fake = gen(z, yb).detach()
                d_real = critic(real, yb).mean()
                d_fake = critic(fake, yb).mean()
                gp = gradient_penalty(
                    critic, real=real, fake=fake, y=yb, device=device, lambda_gp=args.lambda_gp
                )
                d_loss = (d_fake - d_real) + gp
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                step += 1

            z = torch.randn(b, args.z_dim, device=device)
            fake = gen(z, yb)
            g_loss = -critic(fake, yb).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            d_meter.update(float(d_loss.item()), n=b)
            g_meter.update(float(g_loss.item()), n=b)

        print(f"Epoch {epoch}/{args.epochs} - D Loss: {d_meter.avg:.4f} - G Loss: {g_meter.avg:.4f}")

    torch.save(gen.state_dict(), os.path.join(out_dir, "generator.pth"))
    torch.save(critic.state_dict(), os.path.join(out_dir, "critic.pth"))
    with open(os.path.join(out_dir, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump({"persons": persons, "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {os.path.join(out_dir, 'generator.pth')}")


if __name__ == "__main__":
    main()
