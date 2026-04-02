import os
import warnings

import mne
import numpy as np
import scipy.signal as signal

# 屏蔽警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= 路径配置 (自动识别) =================
# 脚本文件所在的目录 (即 DatabaseSpindles 文件夹)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# 父目录 (即 all_datasets 文件夹)
PARENT_DIR = os.path.dirname(DATA_DIR)
# 输出目录 (DatabaseSpindles 的同级目录)
OUTPUT_DIR = os.path.join(PARENT_DIR, "dreams_scoring1_200hz")

print(f"数据读取路径: {DATA_DIR}")
print(f"数据保存路径: {OUTPUT_DIR}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 配置参数
TARGET_FS = 200
WINDOW_SEC = 3
STRIDE_SEC = 1
NEG_KEEP_RATIO = 0.4

# 通道映射
CH_MAP = {
    "excerpt1.edf": "C3-A1",
    "excerpt2.edf": "CZ-A1",
    "excerpt3.edf": "C3-A1",
    "excerpt4.edf": "CZ-A1",
    "excerpt5.edf": "CZ-A1",
    "excerpt6.edf": "CZ-A1",
    "excerpt7.edf": "CZ-A1",
    "excerpt8.edf": "CZ-A1",
}


def get_enhanced_features(data, fs):
    """生成 [T, 3] 增强特征"""
    raw_feat = data
    sos = signal.butter(4, [11, 16], btype="bandpass", fs=fs, output="sos")
    bandpass_feat = signal.sosfiltfilt(sos, data)
    analytic_signal = signal.hilbert(bandpass_feat)
    envelope_feat = np.abs(analytic_signal)
    return np.stack([raw_feat, bandpass_feat, envelope_feat], axis=-1)


def main():
    print(f"开始预处理，目标采样率: {TARGET_FS}Hz")

    for edf_name, ch_name in CH_MAP.items():
        edf_path = os.path.join(DATA_DIR, edf_name)
        if not os.path.exists(edf_path):
            print(f"跳过 {edf_name}: 文件不存在")
            continue

        print(f"\n正在处理: {edf_name} (通道: {ch_name})")

        try:
            # 1. 先不 preload，尝试读取 Header
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

            # 2. 手动将 info 中的日期设为 None，绕过旧版本 MNE 的校验
            with raw.info._unlock():
                raw.info["meas_date"] = None

            # 3. 这时候再加载数据
            raw.load_data()

        except Exception:
            # 如果还是因为日期格式报错，则尝试 preload=True 的读取
            print("  - 正常读取失败，尝试强制绕过 Header 校验...")
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)
            except Exception as e2:
                print(f"  - 无法读取文件 {edf_name}: {e2}")
                continue

        # 获取数据
        current_fs = raw.info["sfreq"]
        # 处理可能的通道名空格
        clean_names = [c.strip() for c in raw.ch_names]
        if ch_name not in clean_names:
            print(f"  - 错误: 找不到通道 {ch_name}")
            continue

        data = raw.get_data(picks=ch_name)[0]

        # 重采样
        if current_fs != TARGET_FS:
            print(f"  - 执行重采样: {current_fs}Hz -> {TARGET_FS}Hz")
            data = mne.filter.resample(data, up=TARGET_FS / current_fs)

        # 基础预处理
        data = signal.detrend(data)
        data = data - np.mean(data)
        data = data / (np.std(data) + 1e-8)

        # 特征增强
        features = get_enhanced_features(data, TARGET_FS)

        # 标签处理（Visual_scoring1）
        label_name = f"Visual_scoring1_{edf_name.replace('.edf', '.txt')}"
        label_path = os.path.join(DATA_DIR, label_name)
        n_samples = len(data)
        labels = np.zeros(n_samples, dtype=np.int8)

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        start_sec, duration = float(parts[0]), float(parts[1])
                        s_idx = int(start_sec * TARGET_FS)
                        e_idx = int((start_sec + duration) * TARGET_FS)
                        labels[s_idx:min(e_idx, n_samples)] = 1
            print("  - 标签加载成功")

        # 滑窗
        win_len = int(WINDOW_SEC * TARGET_FS)
        stride = int(STRIDE_SEC * TARGET_FS)
        num_wins = (n_samples - win_len) // stride + 1
        xs, ys = [], []
        for i in range(num_wins):
            s, e = i * stride, i * stride + win_len
            win_y = labels[s:e]
            if np.any(win_y == 1) or (np.random.rand() < NEG_KEEP_RATIO):
                xs.append(features[s:e])
                ys.append(win_y)

        # 保存
        base_name = edf_name.replace(".edf", "")
        np.save(os.path.join(OUTPUT_DIR, f"{base_name}_X.npy"), np.array(xs, dtype=np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"{base_name}_Y.npy"), np.array(ys, dtype=np.int8))
        print(f"  - 完成! 窗口数: {len(xs)}")

    print(f"\n所有数据处理完毕，保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
