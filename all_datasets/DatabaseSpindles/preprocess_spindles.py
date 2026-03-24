import os
from typing import List, Tuple, Dict

import numpy as np


FS = 256  # DREAMS Spindles excerpt sampling rate (Hz), assumed as per your spec
WINDOW_SEC = 10
STRIDE_SEC = 5
WINDOW_LEN = WINDOW_SEC * FS  # 2560
STRIDE = STRIDE_SEC * FS      # 1280

# 基础路径与输出路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dreams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练/验证/测试划分（按 excerpt 划分，仅使用 1-6）
# - 训练：excerpt1-4
# - 验证：excerpt5
# - 测试：excerpt6
TRAIN_EXCERPTS = [1, 2, 3, 4]
VAL_EXCERPTS = [5]
TEST_EXCERPTS = [6]

# 负样本随机下采样概率
NEG_KEEP_PROB = 0.25


def read_signal(excerpt_path: str) -> np.ndarray:
    """
    Step 1: 读取 excerptX.txt，返回一维 EEG 序列 signal: [N]
    约定：
        - 第一行是通道名，如 "[C3-A1]"，跳过
        - 之后每行一个浮点数
    """
    values: List[float] = []
    with open(excerpt_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # 通道名行，例如 "[C3-A1]"
                first = False
                # 简单 sanity check，也可以直接跳过
                continue
            # 数值行
            try:
                values.append(float(line))
            except ValueError:
                # 如果有意外内容，直接跳过该行
                continue
    signal = np.asarray(values, dtype=np.float32)
    return signal


def read_events_visual_scoring2(label_dir: str, excerpt_id: int) -> List[Tuple[float, float]]:
    """
    Step 2: 读取 Visual_scoring2_excerptX.txt，返回事件列表：[(start_sec, end_sec), ...]

    仅使用专家 2 的标注文件，不再回退到 Visual_scoring1。

    文件格式示例：
        [vis2_Spindles/C3-A1]
          396.5600    1.0000
          422.6900    1.0000
          ...
    第一列：起始时间（秒）
    第二列：持续时间（秒）
    """
    label_path = os.path.join(label_dir, f"Visual_scoring2_excerpt{excerpt_id}.txt")

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Visual_scoring2 file not found for excerpt{excerpt_id}: {label_path}"
        )

    print(f"[excerpt{excerpt_id}] Using expert labels from Visual_scoring2: {os.path.basename(label_path)}")

    events_sec: List[Tuple[float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # 头一行类似 "[vis2_Spindles/C3-A1]"，跳过
                first = False
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                start_sec = float(parts[0])
                duration_sec = float(parts[1])
            except ValueError:
                continue
            end_sec = start_sec + duration_sec
            events_sec.append((start_sec, end_sec))
    return events_sec


def events_to_sample_indices(events_sec: List[Tuple[float, float]], n_samples: int) -> List[Tuple[int, int]]:
    """
    Step 2 (续)：把 (start_sec, end_sec) 换算成 (start_idx, end_idx) 采样点索引区间。
    规则：
        start_idx = floor(start_sec * FS)
        end_idx   = ceil(end_sec * FS)
    并裁剪到 [0, n_samples]。
    """
    events_idx: List[Tuple[int, int]] = []
    for start_sec, end_sec in events_sec:
        start_idx = int(np.floor(start_sec * FS))
        end_idx = int(np.ceil(end_sec * FS))
        # 边界裁剪
        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)
        if end_idx <= start_idx:
            continue
        events_idx.append((start_idx, end_idx))
    return events_idx


def build_pointwise_labels(n_samples: int, events_idx: List[Tuple[int, int]]) -> np.ndarray:
    """
    Step 3: 从事件级标注构造逐点 0/1 标签。
        label: [N]，0 表示非 spindle，1 表示 spindle。
    """
    label = np.zeros(n_samples, dtype=np.int8)
    for start_idx, end_idx in events_idx:
        label[start_idx:end_idx] = 1
    return label


def sliding_windows(
    signal: np.ndarray,
    label: np.ndarray,
    excerpt_id: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Step 4: 对整段信号做滑窗，生成很多 (x_win, y_win) 样本。

    - window_len = 10 秒 = 2560 点
    - stride = 5 秒 = 1280 点
    - 丢弃尾部不足一个完整窗口的部分
    - 负样本窗口随机下采样（NEG_KEEP_PROB）
    - 返回：
        X: [num_win, T, 1]
        Y: [num_win, T]
        meta: 简单元信息，目前包含：
            - 'excerpt_id': [num_win]
            - 'start_idx': [num_win]
            - 'has_spindle': [num_win] (0/1)
    """
    assert signal.shape[0] == label.shape[0], "signal 和 label 长度必须一致"
    n = signal.shape[0]

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    excerpt_ids: List[int] = []
    start_indices: List[int] = []
    has_spindle_flags: List[int] = []

    start = 0
    while start + WINDOW_LEN <= n:
        end = start + WINDOW_LEN
        x_win = signal[start:end]          # [T]
        y_win = label[start:end]           # [T]

        has_spindle = int(np.any(y_win == 1))

        if has_spindle:
            keep = True
        else:
            # 负样本：按概率保留一部分
            keep = np.random.rand() < NEG_KEEP_PROB

        if keep:
            xs.append(x_win[:, None].astype(np.float32))  # [T, 1]
            ys.append(y_win.astype(np.int8))              # [T]
            excerpt_ids.append(excerpt_id)
            start_indices.append(start)
            has_spindle_flags.append(has_spindle)

        start += STRIDE

    if not xs:
        return (
            np.empty((0, WINDOW_LEN, 1), dtype=np.float32),
            np.empty((0, WINDOW_LEN), dtype=np.int8),
            {
                "excerpt_id": np.empty((0,), dtype=np.int32),
                "start_idx": np.empty((0,), dtype=np.int32),
                "has_spindle": np.empty((0,), dtype=np.int8),
            },
        )

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    meta = {
        "excerpt_id": np.asarray(excerpt_ids, dtype=np.int32),
        "start_idx": np.asarray(start_indices, dtype=np.int32),
        "has_spindle": np.asarray(has_spindle_flags, dtype=np.int8),
    }
    return X, Y, meta


def process_single_excerpt(excerpt_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个 excerpt（这里仅用 1~6）做 Step 1-3：
        - 读 EEG 序列
        - 读 Visual_scoring2 事件
        - 生成逐点标签
        - 并把整段 signal/label 存成 npy（保存到 OUTPUT_DIR）
    返回：
        signal: [N]
        label:  [N]
    """
    excerpt_txt = os.path.join(BASE_DIR, f"excerpt{excerpt_id}.txt")
    if not os.path.exists(excerpt_txt):
        raise FileNotFoundError(f"EEG file not found: {excerpt_txt}")

    print(f"Processing excerpt{excerpt_id} ...")
    signal = read_signal(excerpt_txt)  # [N]
    events_sec = read_events_visual_scoring2(BASE_DIR, excerpt_id)  # [(start_sec, end_sec)]
    events_idx = events_to_sample_indices(events_sec, n_samples=signal.shape[0])
    label = build_pointwise_labels(signal.shape[0], events_idx)  # [N]

    # 保存整段信号和标签
    np.save(os.path.join(OUTPUT_DIR, f"excerpt{excerpt_id}_signal.npy"), signal)
    np.save(os.path.join(OUTPUT_DIR, f"excerpt{excerpt_id}_label.npy"), label)

    return signal, label


def main():
    """
    完整预处理流水线：
        1. 对 excerpt1-6 逐个执行：
            - 读 EEG
            - 读 Visual_scoring2 事件
            - 转为逐点 0/1 标签
            - 保存成 excerptX_signal.npy / excerptX_label.npy（保存到 OUTPUT_DIR）
        2. 用 10 秒窗 + 5 秒滑动步长做滑窗：
            - 丢弃最后不足一整窗的部分
            - 全部保留正样本窗
            - 负样本窗按 NEG_KEEP_PROB 随机下采样
        3. 按 excerpt 做 train/val/test 划分：
            - train: excerpt1-4
            - val:   excerpt5
            - test:  excerpt6
        4. 将滑窗样本保存为：
            - windows_train_x.npy / windows_train_y.npy
            - windows_val_x.npy   / windows_val_y.npy
            - windows_test_x.npy  / windows_test_y.npy
        5. 同时保存 meta 信息：
            - meta_train.npz / meta_val.npz / meta_test.npz
    """
    np.random.seed(2024)  # 控制负样本下采样的随机性

    # 用于存放三个子集的滑窗样本
    subset_X = {"train": [], "val": [], "test": []}
    subset_Y = {"train": [], "val": [], "test": []}
    subset_meta = {"train": [], "val": [], "test": []}

    # 仅使用 excerpt1-6（因为 Visual_scoring2 只覆盖到 6）
    for excerpt_id in range(1, 7):
        signal, label = process_single_excerpt(excerpt_id)

        X_win, Y_win, meta = sliding_windows(signal, label, excerpt_id)
        print(
            f"excerpt{excerpt_id}: signal_len={signal.shape[0]}, "
            f"num_windows={X_win.shape[0]}"
        )

        if excerpt_id in TRAIN_EXCERPTS:
            subset = "train"
        elif excerpt_id in VAL_EXCERPTS:
            subset = "val"
        elif excerpt_id in TEST_EXCERPTS:
            subset = "test"
        else:
            continue

        subset_X[subset].append(X_win)
        subset_Y[subset].append(Y_win)
        subset_meta[subset].append(meta)

    # 拼接并保存三个子集
    for subset in ["train", "val", "test"]:
        if not subset_X[subset]:
            print(f"[WARN] No windows collected for subset '{subset}'.")
            X_all = np.empty((0, WINDOW_LEN, 1), dtype=np.float32)
            Y_all = np.empty((0, WINDOW_LEN), dtype=np.int8)
            meta_all = {
                "excerpt_id": np.empty((0,), dtype=np.int32),
                "start_idx": np.empty((0,), dtype=np.int32),
                "has_spindle": np.empty((0,), dtype=np.int8),
            }
        else:
            X_all = np.concatenate(subset_X[subset], axis=0)
            Y_all = np.concatenate(subset_Y[subset], axis=0)

            # 合并 meta
            excerpt_ids = np.concatenate(
                [m["excerpt_id"] for m in subset_meta[subset]], axis=0
            )
            start_idx = np.concatenate(
                [m["start_idx"] for m in subset_meta[subset]], axis=0
            )
            has_spindle = np.concatenate(
                [m["has_spindle"] for m in subset_meta[subset]], axis=0
            )
            meta_all = {
                "excerpt_id": excerpt_ids,
                "start_idx": start_idx,
                "has_spindle": has_spindle,
            }

        np.save(os.path.join(OUTPUT_DIR, f"windows_{subset}_x.npy"), X_all)
        np.save(os.path.join(OUTPUT_DIR, f"windows_{subset}_y.npy"), Y_all)
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"meta_{subset}.npz"),
            excerpt_id=meta_all["excerpt_id"],
            start_idx=meta_all["start_idx"],
            has_spindle=meta_all["has_spindle"],
        )

        print(
            f"Subset '{subset}': X shape = {X_all.shape}, Y shape = {Y_all.shape}, "
            f"saved to windows_{subset}_x.npy / windows_{subset}_y.npy"
        )


if __name__ == "__main__":
    main()

