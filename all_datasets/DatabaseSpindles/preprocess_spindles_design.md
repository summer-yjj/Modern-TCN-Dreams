## DREAMS DatabaseSpindles 预处理设计说明

本设计文档描述了如何将 **DREAMS Spindles Database** 中的原始事件标注与 EEG 信号，转换成适合 **ModernTCN** 训练的 **窗口级逐点二分类样本**。

整体实现对应脚本：`preprocess_spindles.py`。

---

### 1. 关键信息与固定选择

- **使用的标签版本**：  
  - 只使用 `Visual_scoring2_excerptX.txt`（Expert 2 标注）。  
  
- **使用的通道**：  
  - 只使用 `excerptX.txt` 中的单通道 `[C3-A1]`。
- **采样率**：  
  - 固定假设为 **256 Hz**（每秒 256 个采样点）。
- **窗口长度**：  
  - **10 秒**窗口，长度 \(T = 10 \times 256 = 2560\) 个采样点。
- **滑动步长**：  
  - **5 秒**步长，长度 \( \text{stride} = 5 \times 256 = 1280\) 个采样点（窗口间 50% 重叠）。
- **数据集划分方式（按 excerpt 划分）**：  
  - 训练集（train）：excerpt1–excerpt4  
  - 验证集（val）：excerpt5  
  - 测试集（test）：excerpt6  
  - **不**在窗口级别随机划分，防止信息泄漏。
- **标签类型**：  
  - 二分类逐点标签：0 表示非 spindle，1 表示 spindle。

---

### 2. 预处理四步总览

整体链路为：

1. **Step 1：读取原始 EEG 信号**
2. **Step 2：读取事件级 spindle 标注并转为采样点区间**
3. **Step 3：生成逐点 0/1 标签**
4. **Step 4：滑窗切片，得到窗口级训练样本**

---

### 3. Step 1：读取 excerptX.txt 得到连续 EEG 序列

- 输入文件示例：`excerpt1.txt`
- 格式：
  - 第 1 行：通道名，如 `"[C3-A1]"`。
  - 第 2 行及之后：每行一个 EEG 采样值（浮点数）。
- 处理：
  - 跳过首行通道名。
  - 将之后每行转换为 `float32`，收集到列表中。
  - 最终构造一维数组：
    - `signal: np.ndarray`，形状 `[N]`，`N` 为该 excerpt 的采样点数。

---

### 4. Step 2：读取 Visual_scoring2 事件并转为采样点区间

- 标签文件示例：`Visual_scoring2_excerpt1.txt`
- 格式：
  - 第 1 行：头信息，如 `"[vis2_Spindles/C3-A1]"`。
  - 之后每行包含两列数值：
    - 第一列：spindle 起始时间（秒）`start_sec`
    - 第二列：持续时间（秒）`duration_sec`
- 处理流程：
  1. 选 `Visual_scoring2_excerptX.txt`；
  2. 对每条记录计算：
     - `end_sec = start_sec + duration_sec`
  3. 将时间转为采样点索引（统一规则）：
     - `start_idx = floor(start_sec * FS)`
     - `end_idx   = ceil(end_sec * FS)`
  4. 边界裁剪到 `[0, N]`，若 `end_idx <= start_idx` 则丢弃该事件。
- 输出：
  - `events_idx: List[(start_idx, end_idx)]`，每个区间表示在 `[start_idx, end_idx)` 范围内为 spindle。

---

### 5. Step 3：生成逐点 0/1 标签

- 输入：
  - `signal: [N]`
  - `events_idx: [(s1, e1), (s2, e2), ...]`
- 处理：
  - 初始化标签数组：`label = np.zeros(N, dtype=np.int8)`。
  - 对于每个 `(start_idx, end_idx)`：
    - 执行 `label[start_idx:end_idx] = 1`
  - 重叠事件自动合并为连续的 1 区间。
- 输出：
  - `label: [N]`，逐点二分类标签。
- 保存：
  - 对每个 excerptX：
    - `excerptX_signal.npy`
    - `excerptX_label.npy`

---

### 6. Step 4：滑窗切片与样本筛选

1. **滑窗参数**：
   - 窗长：`WINDOW_LEN = 10s * 256Hz = 2560`
   - 步长：`STRIDE = 5s * 256Hz = 1280`
2. **切窗方式**：
   - 从 `start = 0` 开始，依次取：
     - `end = start + WINDOW_LEN`
     - 若 `end > N`，停止，丢弃尾部不足一个整窗的部分。
   - 每个窗口：
     - `x_win = signal[start:end]`  → `[T]`
     - `y_win = label[start:end]`   → `[T]`
     - 再将输入扩展单通道：
       - `x_win` → `[T, 1]`
3. **正/负样本定义**：
   - 若 `y_win` 中存在任意 1，则为 **正样本窗**。
   - 若全部为 0，则为 **负样本窗**。
4. **负样本下采样**：
   - 全部保留正样本窗。
   - 负样本窗以固定概率 `NEG_KEEP_PROB`（脚本中默认 0.25）随机保留。
5. **元信息记录**：
   - 对每个窗口记录：
     - `excerpt_id`：来自哪一段 excerpt（1–6）。
     - `start_idx`：该窗口在原始整段中的起始采样点索引。
     - `has_spindle`：该窗口是否包含 spindle（0/1）。

---

### 7. Train/Val/Test 划分与最终保存格式

- 按 excerpt 划分：
  - 训练集（train）：excerpt1–4
  - 验证集（val）：excerpt5
  - 测试集（test）：excerpt6
- 不在窗口级别随机划分，避免同一 excerpt 的不同窗口出现在不同子集中，防止信息泄漏。

#### 7.1 整段级别保存

对于每个 excerpt\(X\)：

- `excerptX_signal.npy`：整段 EEG 序列 `[N]`
- `excerptX_label.npy`：整段逐点标签 `[N]`

#### 7.2 窗口级别保存

执行完全部预处理与滑窗后，脚本会为三个子集分别保存：

- 训练集：
  - `windows_train_x.npy`：形状 `[num_train_windows, T, 1]`
  - `windows_train_y.npy`：形状 `[num_train_windows, T]`
  - `meta_train.npz`：
    - `excerpt_id`：每个窗口所属 excerpt ID
    - `start_idx`：窗口在整段中的起点索引
    - `has_spindle`：0/1，是否包含 spindle
- 验证集：
  - `windows_val_x.npy` / `windows_val_y.npy`
  - `meta_val.npz`（字段同上）
- 测试集：
  - `windows_test_x.npy` / `windows_test_y.npy`
  - `meta_test.npz`（字段同上）

---

### 8. 总结

通过上述四步预处理，我们完成了：

1. 从原始 `excerptX.txt` 中抽取单导联 EEG 序列 `[N]`。
2. 从 `Visual_scoring2`中读取事件级 spindle 标注，统一转为采样点区间。
3. 将事件级标注变换为逐点 0/1 标签数组，与 EEG 一一对齐。
4. 按 10 秒窗 + 5 秒步长做滑窗，得到大量窗口级样本 `[T,1]` / `[T]`，并按 excerpt 划分为 train/val/test。

这些输出可以直接作为 ModernTCN 或其他序列模型的输入，用于 **逐点 spindle 检测的二分类训练**。

