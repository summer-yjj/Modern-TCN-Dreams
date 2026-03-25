import os
from typing import List, Tuple, Dict

import numpy as np


FS = 256  # DREAMS Spindles excerpt sampling rate (Hz)
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
                first = False
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    signal = np.asarray(values, dtype=np.float32)
    return signal


def read_events_from_file(label_path: str) -> List[Tuple[float, float]]:
    """
    读取单个 Visual_scoring 文件，返回事件列表：[(start_sec, end_sec), ...]
    文件每行格式：start_sec duration_sec
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    events_sec: List[Tuple[float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # 第一行为头信息，例如 [vis1_Spindles/C3-A1]
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
            if end_sec > start_sec:
                events_sec.append((start_sec, end_sec))
    return events_sec


def merge_intervals_union(events_sec: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    对已经得到的最终事件列表做区间合并：
    - 有重叠区间则合并
    - 合并规则对应“最早起点 + 最晚终点”
    注意：
        这个函数只负责“最终事件表”的去重/合并，
        不负责两个专家之间的事件匹配。
    """
    if not events_sec:
        return []

    events_sorted = sorted(events_sec, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = events_sorted[0]

    for s, e in events_sorted[1:]:
        if s <= cur_e:
            # 区间重叠，取最晚终点
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged




def events_match(
    event1: Tuple[float, float],
    event2: Tuple[float, float],
    center_tol_sec: float = 0.2,
) -> bool:
    """
    判断两个专家的两个事件是否可视为“同一个纺锤波”。
    匹配规则尽量保持简单稳健：
    1) 两个区间有重叠 -> 匹配
    2) 或者两个事件中心点距离 <= center_tol_sec -> 匹配
    """
    s1, e1 = event1
    s2, e2 = event2

    overlap = min(e1, e2) - max(s1, s2)
    if overlap > 0:
        return True

    c1 = 0.5 * (s1 + e1)
    c2 = 0.5 * (s2 + e2)
    return abs(c1 - c2) <= center_tol_sec


def match_event_pairs(
    events_1: List[Tuple[float, float]],
    events_2: List[Tuple[float, float]],
    center_tol_sec: float = 0.2,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    只保留“两个专家都标到”的事件对。
    使用双指针做一对一向前匹配，避免一个事件被重复匹配多次。
    """
    events_1 = sorted(events_1, key=lambda x: x[0])
    events_2 = sorted(events_2, key=lambda x: x[0])

    i, j = 0, 0
    matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    while i < len(events_1) and j < len(events_2):
        ev1 = events_1[i]
        ev2 = events_2[j]

        if events_match(ev1, ev2, center_tol_sec=center_tol_sec):
            matched_pairs.append((ev1, ev2))
            i += 1
            j += 1
            continue

        # 没匹配成功时，让结束更早的那个事件先向后移动
        if ev1[1] < ev2[1]:
            i += 1
        else:
            j += 1

    return matched_pairs


def build_consensus_events_from_pairs(
    matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[Tuple[float, float]]:
    """
    对匹配成功的事件对，取“最早起点 + 最晚终点”作为最终正类事件。
    然后再做一次区间合并，消除可能出现的轻微重叠。
    """
    consensus_events: List[Tuple[float, float]] = []
    for (s1, e1), (s2, e2) in matched_pairs:
        consensus_events.append((min(s1, s2), max(e1, e2)))

    return merge_intervals_union(consensus_events)


def read_events_union_visual_scoring12(label_dir: str, excerpt_id: int) -> List[Tuple[float, float]]:
    """
    Step 2: 同时读取 Visual_scoring1 和 Visual_scoring2，
    只保留“两个专家都标到”的事件：
    - 先做事件级匹配
    - 只保留匹配成功的事件对
    - 对每一对事件取最早起点 + 最晚终点
    其余流程保持不变。
    """
    path1 = os.path.join(label_dir, f"Visual_scoring1_excerpt{excerpt_id}.txt")
    path2 = os.path.join(label_dir, f"Visual_scoring2_excerpt{excerpt_id}.txt")

    events_1 = read_events_from_file(path1)
    events_2 = read_events_from_file(path2)

    matched_pairs = match_event_pairs(events_1, events_2, center_tol_sec=0.2)
    union_events = build_consensus_events_from_pairs(matched_pairs)

    print(
        f"[excerpt{excerpt_id}] matched labels from "
        f"{os.path.basename(path1)} + {os.path.basename(path2)} | "
        f"vis1={len(events_1)}, vis2={len(events_2)}, "
        f"matched_pairs={len(matched_pairs)}, consensus={len(union_events)}"
    )
    return union_events


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
    其余策略保持与原脚本一致。
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
    对单个 excerpt（1~6）执行：
        - 读 EEG 序列
        - 读 Visual_scoring1 + Visual_scoring2，并取并集
        - 生成逐点标签
        - 保存整段 signal/label 到 OUTPUT_DIR
    """
    excerpt_txt = os.path.join(BASE_DIR, f"excerpt{excerpt_id}.txt")
    if not os.path.exists(excerpt_txt):
        raise FileNotFoundError(f"EEG file not found: {excerpt_txt}")

    print(f"Processing excerpt{excerpt_id} ...")
    signal = read_signal(excerpt_txt)  # [N]
    events_sec = read_events_union_visual_scoring12(BASE_DIR, excerpt_id)
    events_idx = events_to_sample_indices(events_sec, n_samples=signal.shape[0])
    label = build_pointwise_labels(signal.shape[0], events_idx)  # [N]

    np.save(os.path.join(OUTPUT_DIR, f"excerpt{excerpt_id}_signal.npy"), signal)
    np.save(os.path.join(OUTPUT_DIR, f"excerpt{excerpt_id}_label.npy"), label)
    return signal, label


def main():
    """
    完整预处理流水线（与原脚本一致），唯一差异：
    标注来源改为 Visual_scoring1 + Visual_scoring2 的并集。
    生成输出目录仍为 Dreams。
    """
    np.random.seed(2024)

    subset_X = {"train": [], "val": [], "test": []}
    subset_Y = {"train": [], "val": [], "test": []}
    subset_meta = {"train": [], "val": [], "test": []}

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

