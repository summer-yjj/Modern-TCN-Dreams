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


def clip_events_to_range(events: List[Tuple[float, float]], start: float, end: float) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for s, e in events:
        left = max(s, start)
        right = min(e, end)
        if right > left:
            clipped.append((left, right))
    return clipped


def plot_detail_window(
    excerpt_id: int,
    t: np.ndarray,
    signal: np.ndarray,
    expert1: List[Tuple[float, float]],
    expert2: List[Tuple[float, float]],
    overlap_unions: List[Tuple[float, float]],
    start: float,
    end: float,
    save_path: str,
):
    """
    单张细节图（3行）：
    1) EEG + overlap-union（红色）+ 两专家起止线
    2) Expert1 事件条
    3) Expert2 事件条
    """
    idx = np.where((t >= start) & (t <= end))[0]
    if idx.size == 0:
        return

    local_t = t[idx]
    local_x = signal[idx]

    e1_local = clip_events_to_range(expert1, start, end)
    e2_local = clip_events_to_range(expert2, start, end)
    unions_local = clip_events_to_range(overlap_unions, start, end)

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    ax0 = axes[0]
    ax0.plot(local_t, local_x, color="black", linewidth=0.8)
    for s, e in unions_local:
        ax0.axvspan(s, e, color="red", alpha=0.22)

    for s, e in e1_local:
        ax0.axvline(s, color="#1f77b4", linewidth=0.8, alpha=0.55)
        ax0.axvline(e, color="#1f77b4", linewidth=0.8, alpha=0.55, linestyle="--")
    for s, e in e2_local:
        ax0.axvline(s, color="#2ca02c", linewidth=0.8, alpha=0.55)
        ax0.axvline(e, color="#2ca02c", linewidth=0.8, alpha=0.55, linestyle="--")

    ax0.set_ylabel("EEG")
    ax0.set_title(
        f"Excerpt {excerpt_id} | detail window [{start:.2f}s, {end:.2f}s] (show spindle morphology)",
        fontsize=11,
    )

    draw_event_axis(axes[1], e1_local, "#1f77b4", "Expert 1")
    draw_event_axis(axes[2], e2_local, "#2ca02c", "Expert 2")
    axes[2].set_xlabel("Time (s)")

    line_eeg = plt.Line2D([], [], color="black", linewidth=1.0, label="EEG")
    patch_overlap = plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.22, label="Overlap-triggered union")
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

    for ax in axes:
        ax.set_xlim(start, end)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


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
    parser.add_argument(
        "--detail_window_sec",
        type=float,
        default=3.0,
        help="Window size (seconds) for detailed plots to inspect spindle morphology.",
    )
    parser.add_argument(
        "--max_detail_per_excerpt",
        type=int,
        default=12,
        help="Max number of detailed windows saved for each excerpt.",
    )
    parser.add_argument(
        "--detail_output_dir",
        type=str,
        default="expert_overlap_detail_3s",
        help="Directory to save detailed per-window figures.",
    )
    args = parser.parse_args()

    n = args.num_excerpts
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.0 * n), squeeze=False)
    detail_dir = args.detail_output_dir
    if not os.path.isabs(detail_dir):
        detail_dir = os.path.join(args.data_dir, detail_dir)
    os.makedirs(detail_dir, exist_ok=True)

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

        # 细化图：围绕 overlap-union 的中心，截取 3s EEG，便于观察纺锤波形状
        half = 0.5 * float(args.detail_window_sec)
        detailed = 0
        for k, (s, e) in enumerate(overlap_unions):
            if detailed >= int(args.max_detail_per_excerpt):
                break
            center = 0.5 * (s + e)
            start = max(float(t[0]), center - half)
            end = min(float(t[-1]), center + half)
            if end - start < 1e-6:
                continue

            save_path = os.path.join(
                detail_dir,
                f"excerpt{i:02d}_detail_{k:03d}_{start:.2f}s_{end:.2f}s.png",
            )
            plot_detail_window(
                excerpt_id=i,
                t=t,
                signal=signal,
                expert1=expert1,
                expert2=expert2,
                overlap_unions=overlap_unions,
                start=start,
                end=end,
                save_path=save_path,
            )
            detailed += 1

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
    print(f"saved detail windows in: {detail_dir}")


if __name__ == "__main__":
    main()
