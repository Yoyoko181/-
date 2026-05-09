import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _list_person_mat_files(interim_subset_dir: str) -> List[str]:
    if not os.path.isdir(interim_subset_dir):
        return []
    out: List[str] = []
    for fn in os.listdir(interim_subset_dir):
        if fn.endswith(".mat") and fn != "person_names.mat":
            out.append(os.path.join(interim_subset_dir, fn))
    out.sort(key=lambda p: _person_num(os.path.basename(p)))
    return out


def _person_num(name: str) -> int:
    m = re.match(r"^P(\d+)\.mat$", name)
    if m:
        return int(m.group(1))
    return 10**9


def _load_footstep_feat(mat_path: str) -> np.ndarray:
    d = sio.loadmat(mat_path)
    if "footstep_feat" not in d:
        raise KeyError(f"{mat_path} missing footstep_feat")
    x = d["footstep_feat"]
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    return x


@dataclass(frozen=True)
class SignalSampleIndex:
    subset: str
    person: str
    mat_path: str
    row: int


def build_index_from_interim(
    interim_root: str,
    dataset: str,
    subsets: Sequence[str],
    min_events_per_person: int = 1,
) -> Tuple[List[SignalSampleIndex], List[str]]:
    samples: List[SignalSampleIndex] = []
    persons_set = set()
    for subset in subsets:
        subset_dir = os.path.join(interim_root, dataset, subset)
        for mat_path in _list_person_mat_files(subset_dir):
            person = os.path.splitext(os.path.basename(mat_path))[0]
            x = _load_footstep_feat(mat_path)
            if x.shape[0] < min_events_per_person:
                continue
            persons_set.add(person)
            for r in range(x.shape[0]):
                samples.append(
                    SignalSampleIndex(subset=subset, person=person, mat_path=mat_path, row=r)
                )
    persons = sorted(list(persons_set), key=_person_num)
    return samples, persons


def stratified_split_indices(
    labels: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        n_val = max(1, n_val) if len(cls_idx) >= 5 and val_ratio > 0 else n_val
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class FootstepSignalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SignalSampleIndex],
        class_to_idx: Dict[str, int],
        subset_to_idx: Optional[Dict[str, int]] = None,
        normalize: str = "zscore",
        return_domain: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = dict(class_to_idx)
        self.subset_to_idx = dict(subset_to_idx) if subset_to_idx is not None else None
        self.normalize = normalize
        self.return_domain = return_domain
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_mat(self, mat_path: str) -> np.ndarray:
        x = self._cache.get(mat_path)
        if x is None:
            x = _load_footstep_feat(mat_path)
            self._cache[mat_path] = x
        return x

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "zscore":
            m = float(x.mean())
            s = float(x.std())
            if s < 1e-6:
                s = 1.0
            return (x - m) / s
        if self.normalize == "rms":
            rms = float(np.sqrt(np.mean(x * x)))
            if rms < 1e-6:
                rms = 1.0
            return x / rms
        raise ValueError(f"Unknown normalize={self.normalize}")

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        mat = self._get_mat(s.mat_path)
        sig = mat[s.row]
        sig = self._norm(sig.astype(np.float32))
        x = torch.from_numpy(sig).unsqueeze(0)
        y = torch.tensor(self.class_to_idx[s.person], dtype=torch.long)
        if not self.return_domain:
            return x, y
        if self.subset_to_idx is None:
            d = torch.tensor(0, dtype=torch.long)
        else:
            d = torch.tensor(self.subset_to_idx.get(s.subset, 0), dtype=torch.long)
        return x, y, d

