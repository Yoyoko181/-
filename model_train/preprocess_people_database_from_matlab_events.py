#!e:\毕设\Terra-main\model_train\.venv\Scripts\python.exe
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pywt
import scipy.io
from PIL import Image


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


def _iter_person_mats(subset_dir: Path) -> list[Path]:
    if not subset_dir.exists():
        return []
    mats = [p for p in subset_dir.glob("P*.mat") if p.is_file() and re.fullmatch(r"P\d+\.mat", p.name)]
    mats.sort(key=lambda p: int(p.stem[1:]))
    return mats


def _load_footstep_feat(mat_path: Path) -> np.ndarray:
    d = scipy.io.loadmat(str(mat_path))
    if "footstep_feat" not in d:
        keys = [k for k in d.keys() if not k.startswith("__")]
        raise KeyError(f"{mat_path} 缺少 footstep_feat，keys={keys}")
    x = np.asarray(d["footstep_feat"], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{mat_path} footstep_feat 维度异常: {x.shape}")
    return x


def process_subset(
    *,
    interim_subset_dir: Path,
    out_subset_dir: Path,
    image_size: int,
    image_ext: str,
    wavelet: str,
    scales_max: int,
    limit_events_per_person: int,
    force: bool,
) -> None:
    mats = _iter_person_mats(interim_subset_dir)
    if not mats:
        raise FileNotFoundError(f"{interim_subset_dir} 下未找到任何 Pxx.mat")
    scales = np.arange(1, int(scales_max) + 1)
    written = 0
    for mat_path in mats:
        person = mat_path.stem
        out_person = out_subset_dir / person
        out_person.mkdir(parents=True, exist_ok=True)
        if not force and any(out_person.glob(f"*.{image_ext}")):
            continue
        events = _load_footstep_feat(mat_path)
        if limit_events_per_person and limit_events_per_person > 0:
            events = events[:limit_events_per_person]
        for i in range(events.shape[0]):
            sig = events[i].reshape(-1)
            img = _cwt_rgb(sig, scales=scales, wavelet=wavelet, out_size=image_size)
            out_path = out_person / f"{mat_path.stem}__ev{i}.{image_ext}"
            if out_path.exists() and not force:
                continue
            img.save(str(out_path))
            written += 1
    print(f"完成 CWT: {str(interim_subset_dir)} -> {str(out_subset_dir)} images_written={written}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_root", type=str, default=str((Path(__file__).resolve().parents[1] / "people_database" / "interim_matlab").resolve()))
    parser.add_argument("--out_root", type=str, default=str((Path(__file__).resolve().parents[1] / "people_database").resolve()))
    parser.add_argument("--datasets", type=str, default="A1,A2,A3,A4,A5")
    parser.add_argument("--subsets", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_ext", type=str, default="jpg")
    parser.add_argument("--wavelet", type=str, default="morl")
    parser.add_argument("--scales_max", type=int, default=256)
    parser.add_argument("--limit_events_per_person", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    interim_root = Path(args.interim_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("datasets 不能为空")
    subsets_override = [s.strip() for s in args.subsets.split(",") if s.strip()]

    for ds in datasets:
        ds_dir = interim_root / ds
        if not ds_dir.exists():
            continue
        subsets = subsets_override if subsets_override else sorted([p.name for p in ds_dir.iterdir() if p.is_dir()])
        for ss in subsets:
            interim_subset_dir = ds_dir / ss
            if not interim_subset_dir.exists():
                continue
            out_subset_dir = out_root / ds / ss
            process_subset(
                interim_subset_dir=interim_subset_dir,
                out_subset_dir=out_subset_dir,
                image_size=args.image_size,
                image_ext=args.image_ext,
                wavelet=args.wavelet,
                scales_max=args.scales_max,
                limit_events_per_person=args.limit_events_per_person,
                force=bool(args.force),
            )


if __name__ == "__main__":
    main()

