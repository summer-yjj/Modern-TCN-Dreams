import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _binary_events(binary_1d: np.ndarray):
    arr = binary_1d.astype(int)
    events = []
    in_evt = False
    s = 0
    for i, v in enumerate(arr):
        if v == 1 and not in_evt:
            in_evt = True
            s = i
        elif v == 0 and in_evt:
            in_evt = False
            events.append((s, i - 1))
    if in_evt:
        events.append((s, len(arr) - 1))
    return events


def plot_training_curves(history: dict, save_dir: str):
    _ensure_dir(save_dir)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    # Train / Val loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#1f77b4")
    ax.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#ff7f0e")
    ax.set_title("Train Loss / Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "train_val_loss.png"), dpi=160)
    plt.close(fig)

    # Val point-wise F1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get("val_point_f1", []), label="Val Point-wise F1", color="#2ca02c")
    ax.set_title("Val Point-wise F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_point_f1.png"), dpi=160)
    plt.close(fig)

    # Val event-level F1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get("val_event_f1", []), label="Val Event-level F1", color="#d62728")
    ax.set_title("Val Event-level F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_event_f1.png"), dpi=160)
    plt.close(fig)

    # Val event precision / recall
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get("val_event_precision", []), label="Val Event Precision", color="#9467bd")
    ax.plot(epochs, history.get("val_event_recall", []), label="Val Event Recall", color="#8c564b")
    ax.set_title("Val Event Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_event_precision_recall.png"), dpi=160)
    plt.close(fig)

    # Accuracy (aux)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get("val_point_acc", []), label="Val Point Accuracy", color="#17becf")
    ax.set_title("Accuracy (Aux)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_accuracy_aux.png"), dpi=160)
    plt.close(fig)


def plot_point_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Point-wise Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def plot_point_vs_event_metrics(point_metrics: dict, event_metrics: dict, save_path: str):
    labels = ["Precision", "Recall", "F1"]
    point_vals = [point_metrics.get("precision", 0.0), point_metrics.get("recall", 0.0), point_metrics.get("f1", 0.0)]
    event_vals = [event_metrics.get("precision", 0.0), event_metrics.get("recall", 0.0), event_metrics.get("f1", 0.0)]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, point_vals, width=w, label="Point-wise")
    ax.bar(x + w / 2, event_vals, width=w, label="Event-level")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_title("Point-wise vs Event-level Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def plot_event_metric_bars(event_metrics: dict, save_path: str):
    labels = ["Precision", "Recall", "F1"]
    vals = [event_metrics.get("precision", 0.0), event_metrics.get("recall", 0.0), event_metrics.get("f1", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1)
    ax.set_title("Final Event-level Metrics")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def plot_eeg_expert_prediction_panel(
    eeg: np.ndarray,
    expert2: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    fs: float = 256.0,
    expert1: np.ndarray = None,
    title: str = "",
):
    """
    四行图：EEG + Expert1 + Expert2 + Prediction
    - 统一时间轴
    - 标注行画色块
    - EEG 顶部并高亮预测段波形
    """
    T = len(eeg)
    t = np.arange(T) / float(fs)
    e1 = np.zeros(T, dtype=np.int64) if expert1 is None else expert1.astype(np.int64)
    e2 = expert2.astype(np.int64)
    pd = pred.astype(np.int64)

    fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # 1) EEG
    axes[0].plot(t, eeg, color="black", linewidth=0.8, label="EEG")
    # 精细边界高亮：只在预测事件边界内将波形标红，并绘制边界线
    pred_events = _binary_events(pd)
    for s, e in pred_events:
        s = max(0, int(s))
        e = min(T - 1, int(e))
        if e < s:
            continue
        axes[0].plot(t[s:e + 1], eeg[s:e + 1], color="#d62728", linewidth=1.4)
        axes[0].axvspan(t[s], t[e], color="#d62728", alpha=0.12)
        axes[0].axvline(t[s], color="#d62728", linestyle="--", linewidth=0.8, alpha=0.9)
        axes[0].axvline(t[e], color="#d62728", linestyle="--", linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel("EEG")
    axes[0].legend(["EEG", "Predicted spindle segment"], loc="upper right")
    axes[0].set_title(title if title else "EEG + Expert1 + Expert2 + Prediction")

    def draw_event_row(ax, binary, label, color):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=30, va="center")
        for s, e in _binary_events(binary):
            ax.axvspan(s / fs, (e + 1) / fs, color=color, alpha=0.8)

    # 固定颜色
    draw_event_row(axes[1], e1, "Expert1", "#1f77b4")
    draw_event_row(axes[2], e2, "Expert2", "#2ca02c")
    draw_event_row(axes[3], pd, "Pred", "#d62728")
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
