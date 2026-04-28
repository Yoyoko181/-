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


def _pick_interim_matrix(d: dict) -> np.ndarray:
    best = None
    best_size = -1
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if not isinstance(v, np.ndarray):
            continue
        if v.dtype == object or v.size == 0:
            continue
        if v.ndim != 2:
            continue
        if v.shape[1] < 2:
            continue
        size = int(v.shape[0] * v.shape[1])
        if size > best_size:
            best = v
            best_size = size
    if best is None:
        raise KeyError("未在 .mat 中找到二维数值矩阵变量（例如 shape=(samples,1501)）")
    return best


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


def main() -> None:
    parser = argparse.ArgumentParser()
    default_out_root = str((Path(__file__).resolve().parent / "cwt_images").resolve())
    parser.add_argument("--source_type", type=str, default="raw", choices=["raw", "interim"])
    parser.add_argument("--input_path", type=str, default=r"e:\毕设\Foot_database\A3\A3_1")
    parser.add_argument("--dataset", type=str, default="A3")
    parser.add_argument("--subset", type=str, default="A3_1")
    parser.add_argument("--out_root", type=str, default=default_out_root)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_ext", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--wavelet", type=str, default="morl")
    parser.add_argument("--scales_max", type=int, default=256)
    parser.add_argument("--segment_len", type=int, default=1500)
    parser.add_argument("--segments_per_file", type=int, default=10)
    parser.add_argument("--max_files_per_person", type=int, default=0)
    parser.add_argument("--max_persons", type=int, default=0)
    parser.add_argument("--limit_per_person", type=int, default=0)
    parser.add_argument("--mode", type=str, default="energy_peak", choices=["energy_peak", "random"])
    parser.add_argument("--energy_window", type=int, default=256)
    parser.add_argument("--peak_k", type=float, default=2.5)
    parser.add_argument("--min_peak_distance", type=int, default=1024)
    args = parser.parse_args()

    out_root = Path(args.out_root) / args.dataset / args.subset
    out_root.mkdir(parents=True, exist_ok=True)

    scales = np.arange(1, int(args.scales_max) + 1)

    if args.source_type == "raw":
        subset_dir = Path(args.input_path)
        if not subset_dir.exists():
            raise FileNotFoundError(f"找不到目录: {subset_dir}")
        people = _iter_people_raw(subset_dir)
        if args.max_persons and args.max_persons > 0:
            people = people[: args.max_persons]
        if not people:
            raise RuntimeError(f"{subset_dir} 下未找到任何 Pxx/*.mat")

        files_done = 0
        images_done = 0
        print(f"Source: raw", flush=True)
        print(f"Input: {subset_dir}", flush=True)
        print(f"Output: {out_root}", flush=True)
        for person_name, mats in people:
            mats = mats[: args.max_files_per_person] if args.max_files_per_person and args.max_files_per_person > 0 else mats
            for mat_path in mats:
                files_done += 1
                seed = zlib.adler32(str(mat_path).encode("utf-8")) % 1_000_000
                signal = _load_signal_1d_from_mat(mat_path)
                if args.mode == "energy_peak":
                    segs = _segments_energy_peaks(
                        signal,
                        segment_len=args.segment_len,
                        segments_per_file=args.segments_per_file,
                        seed=seed,
                        energy_window=args.energy_window,
                        peak_k=args.peak_k,
                        min_peak_distance=args.min_peak_distance,
                    )
                    if not segs:
                        segs = _segments_random(signal, args.segment_len, args.segments_per_file, seed)
                else:
                    segs = _segments_random(signal, args.segment_len, args.segments_per_file, seed)

                for i, seg in enumerate(segs):
                    img = _cwt_rgb(seg, scales=scales, wavelet=args.wavelet, out_size=args.image_size)
                    out_path = out_root / person_name / f"{mat_path.stem}__seg{i}.{args.image_ext}"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.exists():
                        continue
                    img.save(str(out_path))
                    images_done += 1

                if files_done % 20 == 0:
                    print(f"Progress: files={files_done} images={images_done}", flush=True)

        print(f"Done: files={files_done} images={images_done}", flush=True)
        return

    mat_path = Path(args.input_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到文件: {mat_path}")
    d = scipy.io.loadmat(str(mat_path))
    mat = _pick_interim_matrix(d)
    if mat.shape[1] < 1501:
        raise ValueError(f"interim 矩阵列数不足 1501: shape={mat.shape}")
    features = np.asarray(mat[:, :1500], dtype=np.float32)
    labels = np.asarray(mat[:, 1500]).reshape(-1)
    if labels.size != features.shape[0]:
        raise ValueError("label 行数与 features 不一致")
    labels_int = labels.astype(np.int64, copy=False)
    if labels_int.min() == 0:
        labels_int = labels_int + 1

    rng = np.random.default_rng(42)
    idxs = np.arange(features.shape[0])
    rng.shuffle(idxs)

    per_person_count: dict[int, int] = {}
    written = 0
    print(f"Source: interim", flush=True)
    print(f"Input: {mat_path}", flush=True)
    print(f"Output: {out_root}", flush=True)

    for idx in idxs.tolist():
        pid = int(labels_int[idx])
        if args.limit_per_person and args.limit_per_person > 0:
            c = per_person_count.get(pid, 0)
            if c >= args.limit_per_person:
                continue
        sig = _normalize_1d(features[idx])
        img = _cwt_rgb(sig, scales=scales, wavelet=args.wavelet, out_size=args.image_size)
        person_name = f"P{pid}"
        out_path = out_root / person_name / f"sample_{idx}.{args.image_ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            continue
        img.save(str(out_path))
        per_person_count[pid] = per_person_count.get(pid, 0) + 1
        written += 1
        if written % 200 == 0:
            print(f"Progress: images={written}", flush=True)

    print(f"Done: images={written}", flush=True)


if __name__ == "__main__":
    main()
