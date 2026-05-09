import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_models.utils import ensure_dir, get_device, seed_everything
from new_models.wgan_gp import ConditionalMLPGenerator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wgan_dir", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_per_class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--out_len", type=int, default=1500)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    wgan_dir = os.path.abspath(args.wgan_dir)
    with open(os.path.join(wgan_dir, "class_map.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    persons: List[str] = m["persons"]
    num_classes = len(persons)

    gen = ConditionalMLPGenerator(
        num_classes=num_classes,
        z_dim=args.z_dim,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        out_len=args.out_len,
    ).to(device)
    gen.load_state_dict(torch.load(os.path.join(wgan_dir, "generator.pth"), map_location=device))
    gen.eval()

    ys = []
    xs = []
    for cls in range(num_classes):
        y = torch.full((args.num_per_class,), cls, dtype=torch.long, device=device)
        z = torch.randn(args.num_per_class, args.z_dim, device=device)
        x = gen(z, y).detach().cpu()
        xs.append(x)
        ys.append(torch.full((args.num_per_class,), cls, dtype=torch.long))

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    perm = torch.randperm(x_all.size(0))
    x_all = x_all[perm]
    y_all = y_all[perm]

    out_path = os.path.abspath(args.out_path)
    ensure_dir(os.path.dirname(out_path))
    torch.save({"x": x_all, "y": y_all, "persons": persons}, out_path)
    print(f"Saved: {out_path} x={tuple(x_all.shape)} y={tuple(y_all.shape)}")


if __name__ == "__main__":
    main()
