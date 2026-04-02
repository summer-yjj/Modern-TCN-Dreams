import os
import re
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _extract_excerpt_id(name: str) -> int:
    m = re.search(r"excerpt(\d+)", name)
    if m is None:
        return -1
    return int(m.group(1))


class DREAMSScoring1200HzSegLoader(Dataset):
    """
    读取 preprocess_dreams.py 生成的数据：
      - excerpt1_X.npy / excerpt1_Y.npy
      - ...
      - excerpt8_X.npy / excerpt8_Y.npy

    默认划分：
      - train: excerpt1~6
      - val:   excerpt7
      - test:  excerpt8
    """

    def __init__(self, root_path: str, flag: str = "train"):
        assert flag in ["train", "val", "test"]
        self.root_path = root_path
        self.flag = flag

        split_map: Dict[str, List[int]] = {
            "train": [1, 2, 3, 4, 5, 6],
            "val": [7],
            "test": [8],
        }
        target_ids = set(split_map[flag])

        x_files = sorted(
            [f for f in os.listdir(root_path) if f.endswith("_X.npy") and f.startswith("excerpt")]
        )
        y_files = {f.replace("_Y.npy", ""): f for f in os.listdir(root_path) if f.endswith("_Y.npy")}

        xs, ys = [], []
        loaded_ids = []
        for xf in x_files:
            base = xf.replace("_X.npy", "")
            excerpt_id = _extract_excerpt_id(base)
            if excerpt_id not in target_ids:
                continue
            yf = y_files.get(base)
            if yf is None:
                continue

            x = np.load(os.path.join(root_path, xf))  # [N, T, C]
            y = np.load(os.path.join(root_path, yf))  # [N, T]
            if x.ndim != 3 or y.ndim != 2:
                raise ValueError(f"Bad shape in {xf}/{yf}: x={x.shape}, y={y.shape}")
            if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
                raise ValueError(f"Mismatched x/y in {xf}/{yf}: x={x.shape}, y={y.shape}")

            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.int64))
            loaded_ids.append(excerpt_id)

        if not xs:
            raise FileNotFoundError(
                f"No excerpt*_X/Y.npy found for split='{flag}' in {root_path}. "
                f"Expected excerpt ids: {sorted(list(target_ids))}"
            )

        self.x = np.concatenate(xs, axis=0)  # [N, T, C]
        self.y = np.concatenate(ys, axis=0)  # [N, T]
        self.excerpt_ids = loaded_ids

        pos_ratio = float(self.y.mean()) if self.y.size > 0 else 0.0
        print(
            f"[DREAMSScoring1200HzSegLoader] flag={flag}, excerpts={sorted(loaded_ids)}, "
            f"num_windows={self.x.shape[0]}, window_len={self.x.shape[1]}, "
            f"channels={self.x.shape[2]}, positive_point_ratio={pos_ratio:.6f}"
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_win = self.x[index]  # [T, C]
        y_win = self.y[index]  # [T]
        mask = np.ones_like(y_win, dtype=np.float32)
        return (
            torch.from_numpy(x_win.astype(np.float32)).float(),
            torch.from_numpy(y_win.astype(np.int64)).long(),
            torch.from_numpy(mask).float(),
        )
