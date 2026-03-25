import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_signal(excerpt_path: str) -> np.ndarray:
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
    return np.asarray(values, dtype=np.float32)


def read_events(label_path: str) -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                start = float(parts[0])
                dur = float(parts[1])
            except ValueError:
                continue
            events.append((start, start + dur))
    return events


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


def overlap_union_intervals(
    expert1: List[Tuple[float, float]],
    expert2: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    仅当两个专家事件存在重叠时，才生成区间；
    生成区间取“最早起点 + 最晚终点”（两专家该重叠事件对的并集）。
    """
    unions: List[Tuple[float, float]] = []
    for s1, e1 in expert1:
        for s2, e2 in expert2:
            left = max(s1, s2)
            right = min(e1, e2)
            if right > left:
                unions.append((min(s1, s2), max(e1, e2)))
    return merge_intervals(unions)


def draw_event_axis(ax, events: List[Tuple[float, float]], color: str, title: str):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for s, e in events:
        ax.axvspan(s, e, color=color, alpha=0.45)
        ax.axvline(s, color=color, linestyle="-", linewidth=0.8, alpha=0.8)
        ax.axvline(e, color=color, linestyle="--", linewidth=0.8, alpha=0.8)


def main():
    parser = argparse.ArgumentParser(description="Visualize overlap-based spindle labels for the first 6 excerpts.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path containing excerpt*.txt and Visual_scoring*.txt files.",
    )
    parser.add_argument(
        "--num_excerpts",
        type=int,
        default=6,
        help="How many excerpts to visualize from excerpt1..excerptN.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=256.0,
        help="Sampling frequency for time axis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="expert_overlap_first6.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    n = args.num_excerpts
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.0 * n), squeeze=False)

    for i in range(1, n + 1):
        signal_path = os.path.join(args.data_dir, f"excerpt{i}.txt")
        e1_path = os.path.join(args.data_dir, f"Visual_scoring1_excerpt{i}.txt")
        e2_path = os.path.join(args.data_dir, f"Visual_scoring2_excerpt{i}.txt")

        signal = read_signal(signal_path)
        expert1 = read_events(e1_path)
        expert2 = read_events(e2_path)
        overlap_unions = overlap_union_intervals(expert1, expert2)

        t = np.arange(signal.shape[0]) / float(args.fs)

        # Part 1: EEG + overlap-triggered union spans (red)
        ax0 = axes[i - 1, 0]
        ax0.plot(t, signal, color="black", linewidth=0.6, label="EEG")
        for s, e in overlap_unions:
            ax0.axvspan(s, e, color="red", alpha=0.20)

        # 显示两个专家各自的起止点
        for s, e in expert1:
            ax0.axvline(s, color="#1f77b4", linewidth=0.35, alpha=0.28)
            ax0.axvline(e, color="#1f77b4", linewidth=0.35, alpha=0.28, linestyle="--")
        for s, e in expert2:
            ax0.axvline(s, color="#2ca02c", linewidth=0.35, alpha=0.28)
            ax0.axvline(e, color="#2ca02c", linewidth=0.35, alpha=0.28, linestyle="--")

        ax0.set_title(f"Excerpt {i}: EEG + overlap-based union highlight", fontsize=10)
        ax0.set_ylabel("EEG")

        # Part 2/3: experts
        ax1 = axes[i - 1, 1]
        draw_event_axis(ax1, expert1, "#1f77b4", f"Excerpt {i}: Expert 1")

        ax2 = axes[i - 1, 2]
        draw_event_axis(ax2, expert2, "#2ca02c", f"Excerpt {i}: Expert 2")

        for ax in (ax0, ax1, ax2):
            ax.set_xlim(t[0], t[-1])
            ax.set_xlabel("Time (s)")

    # unified legend
    line_eeg = plt.Line2D([], [], color="black", linewidth=1.0, label="EEG")
    patch_overlap = plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.20, label="Overlap-triggered union")
    e1_start = plt.Line2D([], [], color="#1f77b4", linewidth=1.0, linestyle="-", label="Expert1 start")
    e1_end = plt.Line2D([], [], color="#1f77b4", linewidth=1.0, linestyle="--", label="Expert1 end")
    e2_start = plt.Line2D([], [], color="#2ca02c", linewidth=1.0, linestyle="-", label="Expert2 start")
    e2_end = plt.Line2D([], [], color="#2ca02c", linewidth=1.0, linestyle="--", label="Expert2 end")
    fig.legend(
        handles=[line_eeg, patch_overlap, e1_start, e1_end, e2_start, e2_end],
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.data_dir, out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
