#!e:\毕设\Terra-main\model_train\.venv\Scripts\python.exe
import argparse
import re
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pywt
import scipy.io
from PIL import Image


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


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


def _iter_people_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    people = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"P\d+", p.name)]
    people.sort(key=lambda p: int(p.name[1:]))
    return people


def _iter_images(root: Path) -> list[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]
    out: list[Path] = []
    for pat in exts:
        out.extend(sorted(root.glob(pat)))
    return out


def _default_subsets(dataset: str) -> list[str]:
    ds = dataset.strip().upper()
    if ds == "A2":
        return ["A2_1", "A2_2", "A2_3"]
    if ds == "A3":
        return ["A3_1", "A3_2", "A3_3"]
    if ds == "A5":
        return ["A5_1", "A5_2", "A5_3"]
    return [ds]


def _raw_subset_root(foot_db_root: Path, dataset: str, subset: str) -> Path:
    ds = dataset.strip().upper()
    if ds in {"A2", "A3", "A5"}:
        return foot_db_root / ds / subset
    return foot_db_root / ds


def _out_subset_root(out_root: Path, dataset: str, subset: str) -> Path:
    ds = dataset.strip().upper()
    ss = subset.strip()
    return out_root / ds / ss


def process_subset(
    *,
    foot_db_root: Path,
    out_root: Path,
    dataset: str,
    subset: str,
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
    force: bool,
) -> Path:
    in_root = _raw_subset_root(foot_db_root, dataset, subset)
    if not in_root.exists():
        raise FileNotFoundError(f"找不到 raw 数据目录: {in_root}")

    out_subset = _out_subset_root(out_root, dataset, subset)
    out_subset.mkdir(parents=True, exist_ok=True)

    people_dirs = _iter_people_dirs(in_root)
    if max_persons and max_persons > 0:
        people_dirs = people_dirs[:max_persons]
    if not people_dirs:
        raise RuntimeError(f"{in_root} 下未找到任何 Pxx 目录")

    scales = np.arange(1, int(scales_max) + 1)
    files_done = 0
    images_done = 0
    for person_dir in people_dirs:
        out_person = out_subset / person_dir.name
        if not force and out_person.exists() and _iter_images(out_person):
            continue
        mats = sorted(person_dir.glob("*.mat"))
        if max_files_per_person and max_files_per_person > 0:
            mats = mats[:max_files_per_person]
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
                out_path = out_person / f"{mat_path.stem}__seg{i}.{image_ext}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    continue
                img.save(str(out_path))
                images_done += 1
        if files_done and files_done % 20 == 0:
            print(f"{dataset}/{subset}: files={files_done} images={images_done}", flush=True)
    print(f"完成 {dataset}/{subset}: files_processed={files_done} images_written={images_done}", flush=True)
    return out_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foot_db_root", type=str, default=r"e:\毕设\Foot_database")
    parser.add_argument("--out_root", type=str, default=str((Path(__file__).resolve().parents[1] / "people_database").resolve()))
    parser.add_argument("--datasets", type=str, default="A1,A2,A3,A4,A5")
    parser.add_argument("--subsets", type=str, default="")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    foot_db_root = Path(args.foot_db_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("datasets 不能为空")

    for ds in datasets:
        subsets = [s.strip() for s in args.subsets.split(",") if s.strip()] if str(args.subsets).strip() else _default_subsets(ds)
        for ss in subsets:
            process_subset(
                foot_db_root=foot_db_root,
                out_root=out_root,
                dataset=ds,
                subset=ss,
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
                force=bool(args.force),
            )


if __name__ == "__main__":
    main()
