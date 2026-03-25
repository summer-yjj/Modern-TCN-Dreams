from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = 'none'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: [N, C], target: [N]
        标准 Focal: 基于未加权的 log-softmax 计算 pt，再叠加 alpha_t（class weight）。
        """
        logp = F.log_softmax(logits, dim=-1)                          # [N, C]
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)       # [N]
        pt = logp_t.exp()                                             # [N]
        ce = -logp_t                                                  # [N]

        focal_factor = (1.0 - pt) ** self.gamma

        if self.weight is not None:
            alpha_t = self.weight[target]                             # [N]
            loss = alpha_t * focal_factor * ce
        else:
            loss = focal_factor * ce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalDiceLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, dice_weight: float = 1.0, reduction: str = 'none'):
        super(FocalDiceLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Focal + Dice（二分类）：
        - Focal 部分逐点；
        - Dice 部分为整批标量，再加权到每个样本上。
        """
        # focal 部分
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logp_t.exp()
        ce = -logp_t
        focal_factor = (1.0 - pt) ** self.gamma

        if self.weight is not None:
            alpha_t = self.weight[target]
            focal = alpha_t * focal_factor * ce   # [N]
        else:
            focal = focal_factor * ce             # [N]

        # dice 部分（二分类，针对正类）
        probs = F.softmax(logits, dim=-1)[:, 1]         # [N]
        target_f = target.float()                       # [N]
        eps = 1e-6
        intersection = (probs * target_f).sum()
        denom = probs.sum() + target_f.sum() + eps
        dice_loss = 1.0 - 2.0 * intersection / denom    # 标量

        loss = focal + self.dice_weight * dice_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class Exp_PointSeg(Exp_Basic):
    def __init__(self, args):
        super(Exp_PointSeg, self).__init__(args)

    def _build_model(self):
        # 使用 DREAMSSegLoader 的窗口形状和标签分布来配置模型
        train_data, _ = self._get_data(flag='TRAIN')
        # train_data.x: [N, T, C]
        self.args.seq_len = train_data.x.shape[1]
        self.args.pred_len = 0
        self.args.enc_in = train_data.x.shape[2]  # 通道数，一般为 1
        self.args.num_class = 2  # 逐点二分类：非 spindle / spindle

        # 类别权重：若指定了 pos_weight 且 >0 则用手动权重，否则按训练集自动计算
        if hasattr(self.args, 'pos_weight') and self.args.pos_weight > 0:
            self.class_weights = torch.tensor([1.0, self.args.pos_weight], dtype=torch.float32, device=self.device)
        else:
            w_np = self._compute_class_weights(train_data)
            self.class_weights = torch.tensor(w_np, dtype=torch.float32, device=self.device) if w_np is not None else None

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_balanced_train_loader(self, train_data, fallback_loader):
        """
        使用 WeightedRandomSampler 在窗口级进行正样本过采样，提升训练中正窗口出现频率。
        正窗口定义：窗口中至少存在一个正标签点。
        """
        use_sampling = getattr(self.args, 'pointseg_balanced_sampling', True)
        if not use_sampling:
            return fallback_loader

        labels = getattr(train_data, "y", None)
        if labels is None or len(labels) == 0:
            return fallback_loader

        pos_mask = (labels.sum(axis=1) > 0)
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        n_total = n_pos + n_neg
        if n_total == 0 or n_pos == 0 or n_neg == 0:
            return fallback_loader

        target_pos_ratio = float(getattr(self.args, "pointseg_target_pos_ratio", 0.5))
        target_pos_ratio = min(max(target_pos_ratio, 1e-4), 1 - 1e-4)
        target_neg_ratio = 1.0 - target_pos_ratio

        # 令 E[pos_ratio] ≈ target_pos_ratio
        # 对每个窗口赋权：w_pos / w_neg = (target_pos_ratio / n_pos) / (target_neg_ratio / n_neg)
        w_pos = target_pos_ratio / n_pos
        w_neg = target_neg_ratio / n_neg
        sample_weights = np.where(pos_mask, w_pos, w_neg).astype(np.float64)
        sample_weights = torch.from_numpy(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_total,
            replacement=True,
        )

        print(
            f"[Exp_PointSeg] balanced sampler enabled: "
            f"windows total={n_total}, pos={n_pos}, neg={n_neg}, "
            f"raw_pos_ratio={n_pos / n_total:.4f}, target_pos_ratio={target_pos_ratio:.4f}"
        )

        train_loader = DataLoader(
            train_data,
            batch_size=fallback_loader.batch_size,
            sampler=sampler,
            num_workers=fallback_loader.num_workers,
            drop_last=fallback_loader.drop_last,
        )
        return train_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 根据配置选择损失：ce / focal / focal_dice
        weight = getattr(self, 'class_weights', None)
        loss_type = getattr(self.args, 'pointseg_loss', 'ce').lower()
        if loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
        elif loss_type == 'focal':
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            criterion = FocalLoss(weight=weight, gamma=gamma, reduction='none')
        elif loss_type in ('focal_dice', 'focal+dice'):
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            dice_w = getattr(self.args, 'dice_weight', 1.0)
            criterion = FocalDiceLoss(weight=weight, gamma=gamma, dice_weight=dice_w, reduction='none')
        else:
            raise ValueError(f"Unsupported pointseg_loss: {loss_type}")
        return criterion

    def _compute_class_weights(self, train_data):
        """
        根据训练集逐点标签 (0/1) 的分布，按类别频率的反比构造 CE 的类别权重。
        返回 numpy 数组 [w0, w1]，并保证平均权重约为 1。
        """
        # train_data.y: [N, T] np.ndarray，取值为 0/1
        labels = getattr(train_data, "y", None)
        if labels is None:
            return None
        flat = labels.reshape(-1)
        count0 = np.sum(flat == 0)
        count1 = np.sum(flat == 1)
        total = count0 + count1
        if total == 0 or count0 == 0 or count1 == 0:
            # 极端情况：某一类缺失，退化为均等权重
            return np.array([1.0, 1.0], dtype=np.float32)
        # 经典 inverse frequency：total / (2 * count_c)，使两类平均权重约为 1
        w0 = total / (2.0 * count0) if count0 > 0 else 1.0
        w1 = total / (2.0 * count1)
        class_weights = np.array([w0, w1], dtype=np.float32)
        print(f"[Exp_PointSeg] class weights (0->1): {class_weights}")
        return class_weights

    def _pointwise_loss_and_preds(self, outputs, label, mask, criterion):
        """
        outputs: [B, T, C]
        label:   [B, T]
        mask:    [B, T] (1 表示有效)
        返回：标量 loss、展平后的预测和标签（仅保留 mask=1 的点）
        """
        B, T, C = outputs.shape
        logits = outputs.reshape(-1, C)          # [B*T, C]
        labels_flat = label.reshape(-1)          # [B*T]
        mask_flat = mask.reshape(-1) > 0.5       # [B*T]

        logits = logits[mask_flat]
        labels_flat = labels_flat[mask_flat].long()

        if logits.numel() == 0:
            # 没有有效点时，返回零损失并避免 NaN
            loss = torch.tensor(0.0, device=outputs.device)
            preds_flat = torch.zeros(0, dtype=torch.long, device=outputs.device)
            return loss, preds_flat, labels_flat

        loss_vec = criterion(logits, labels_flat)  # [num_valid_points]
        loss = loss_vec.mean()
        #preds_flat = torch.argmax(logits, dim=-1)  # [num_valid_points]
        probs = torch.softmax(logits, dim=-1)
        pos_probs = probs[:, 1]
        threshold = self.args.pos_threshold if hasattr(self.args, "pos_threshold") else 0.5
        preds_flat = (pos_probs >= threshold).long()
        return loss, preds_flat, labels_flat

    def _pointwise_metrics(self, preds, trues):
        """
        preds, trues: numpy 1D arrays of 0/1 labels on valid points
        返回: dict(acc, precision, recall, f1)
        """
        if preds.size == 0:
            return dict(acc=0.0, precision=0.0, recall=0.0, f1=0.0)
        acc = cal_accuracy(preds, trues)
        tp = np.logical_and(preds == 1, trues == 1).sum()
        fp = np.logical_and(preds == 1, trues == 0).sum()
        fn = np.logical_and(preds == 0, trues == 1).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return dict(acc=float(acc), precision=float(precision), recall=float(recall), f1=float(f1))

    def _binary_to_events(self, binary_1d: np.ndarray):
        """
        把 0/1 序列转换为闭区间事件列表 [(start, end), ...]。
        """
        if binary_1d.size == 0:
            return []
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

    def _overlap(self, e1, e2):
        s1, t1 = e1
        s2, t2 = e2
        left = max(s1, s2)
        right = min(t1, t2)
        return max(0, right - left + 1)

    def _event_match_one_to_one(self, pred_events, true_events):
        """
        一对一重叠匹配：每个预测事件最多匹配一个真实事件，反之亦然。
        判定匹配条件：事件有重叠（overlap > 0）。
        """
        matched_true = set()
        tp = 0
        for pe in pred_events:
            best_j = -1
            best_ov = 0
            for j, te in enumerate(true_events):
                if j in matched_true:
                    continue
                ov = self._overlap(pe, te)
                if ov > best_ov:
                    best_ov = ov
                    best_j = j
            if best_j >= 0 and best_ov > 0:
                tp += 1
                matched_true.add(best_j)
        fp = max(0, len(pred_events) - tp)
        fn = max(0, len(true_events) - tp)
        return tp, fp, fn

    def _event_level_metrics_from_batch(self, outputs, label, mask):
        """
        事件级评估：
        1) 概率阈值二值化
        2) 候选事件提取
        3) 先去除过短事件，再合并间隔过近事件
        4) 一对一重叠匹配计算 TP/FP/FN
        """
        probs = torch.softmax(outputs, dim=-1)[..., 1]  # [B, T]
        pred_bin = (probs >= float(getattr(self.args, "pos_threshold", 0.2))).detach().cpu().numpy().astype(np.int64)
        true_bin = label.detach().cpu().numpy().astype(np.int64)
        mask_np = mask.detach().cpu().numpy().astype(np.int64)

        fs = float(getattr(self.args, "fs", 256.0))
        min_dur = float(getattr(self.args, "event_min_duration_sec", 0.5))
        merge_gap = float(getattr(self.args, "event_merge_gap_sec", 0.15))
        one_to_one = bool(getattr(self.args, "event_one_to_one", True))

        tp_all, fp_all, fn_all = 0, 0, 0
        B = pred_bin.shape[0]
        for b in range(B):
            valid = mask_np[b] > 0
            pred_seq = pred_bin[b][valid]
            true_seq = true_bin[b][valid]

            pred_events = self._postprocess_spindle_events(
                pred_seq,
                np.ones_like(pred_seq, dtype=np.int64),
                fs=fs,
                merge_gap_sec=merge_gap,
                min_dur_sec=min_dur,
                max_dur_sec=999.0,
            )
            true_events = self._binary_to_events(true_seq)

            if one_to_one:
                tp, fp, fn = self._event_match_one_to_one(pred_events, true_events)
            else:
                # 兜底：非一对一时仍以重叠命中为 TP（较宽松）
                tp = 0
                used_true = set()
                for pe in pred_events:
                    for j, te in enumerate(true_events):
                        if self._overlap(pe, te) > 0:
                            tp += 1
                            used_true.add(j)
                            break
                fp = max(0, len(pred_events) - tp)
                fn = max(0, len(true_events) - len(used_true))

            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        recall = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return dict(tp=tp_all, fp=fp_all, fn=fn_all, precision=precision, recall=recall, f1=f1)

    def _postprocess_spindle_events(
        self,
        point_preds_1d: np.ndarray,
        valid_mask_1d: np.ndarray,
        fs: float = 256.0,
        merge_gap_sec: float = 0.2,
        min_dur_sec: float = 0.5,
        max_dur_sec: float = 3.0,
    ):
        """
        将单个窗口的逐点 0/1 预测序列转为纺锤波事件列表，并根据时间规则做合并和过滤。

        规则：
        - 仅保留 valid_mask=1 的有效点；
        - 先找到所有连续的 1 段；
        - 合并相邻且间隔 <= merge_gap_sec 的片段；
        - 删除持续时间 < min_dur_sec 或 > max_dur_sec 的片段。

        返回：List[(start_idx, end_idx)]，索引在窗口内部（按有效点序列计数）。
        """
        if point_preds_1d.size == 0:
            return []

        preds = point_preds_1d.astype(int)
        mask = valid_mask_1d.astype(bool)

        # 只保留有效点
        preds = preds[mask]
        if preds.size == 0:
            return []

        # 找连续的 1 段
        events = []
        T = preds.shape[0]
        in_event = False
        start_idx = 0
        for t in range(T):
            if preds[t] == 1 and not in_event:
                in_event = True
                start_idx = t
            elif preds[t] == 0 and in_event:
                in_event = False
                end_idx = t - 1
                events.append((start_idx, end_idx))
        if in_event:
            events.append((start_idx, T - 1))

        if not events:
            return []

        # 合并间隔 <= merge_gap_sec 的相邻事件
        gap_max = int(np.floor(merge_gap_sec * fs))
        merged = []
        cur_start, cur_end = events[0]
        for s, e in events[1:]:
            gap = s - cur_end - 1
            if gap <= gap_max:
                cur_end = e
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # 根据持续时间过滤
        final_events = []
        for s, e in merged:
            dur_sec = (e - s + 1) / fs
            if min_dur_sec <= dur_sec <= max_dur_sec:
                final_events.append((s, e))

        return final_events

    def _events_to_point_preds(self, events, valid_mask_1d):
        """
        将事件列表还原成按有效点序列计数的 0/1 预测，再映射回原窗口长度。
        events: List[(start_idx, end_idx)]，索引基于有效点序列
        valid_mask_1d: [T]，1 表示有效
        返回: [T] 的 0/1 numpy 数组（无效位置为 0）
        """
        mask = valid_mask_1d.astype(bool)
        out = np.zeros_like(valid_mask_1d, dtype=np.int64)

        valid_len = int(mask.sum())
        if valid_len == 0:
            return out

        valid_pred = np.zeros(valid_len, dtype=np.int64)
        for s, e in events:
            if s < 0 or e < s:
                continue
            s_clamped = max(0, min(s, valid_len - 1))
            e_clamped = max(0, min(e, valid_len - 1))
            valid_pred[s_clamped:e_clamped + 1] = 1

        out[mask] = valid_pred
        return out

    def _pointwise_loss_and_preds_postprocessed(self, outputs, label, mask, criterion):
        """
        在原始逐点预测基础上，增加 spindle 事件后处理，再返回展平后的 preds / labels。
        """
        B, T, C = outputs.shape
        logits = outputs.reshape(-1, C)
        labels_flat_all = label.reshape(-1)
        mask_flat = mask.reshape(-1) > 0.5

        logits_valid = logits[mask_flat]
        labels_valid_all = labels_flat_all[mask_flat].long()

        if logits_valid.numel() == 0:
            loss = torch.tensor(0.0, device=outputs.device)
            preds_valid = torch.zeros(0, dtype=torch.long, device=outputs.device)
            return loss, preds_valid, labels_valid_all

        loss_vec = criterion(logits_valid, labels_valid_all)
        loss = loss_vec.mean()

        # 基于整张 outputs 做后处理
        probs = torch.softmax(outputs, dim=-1)   # [B, T, 2]
        pos_probs = probs[..., 1]                # [B, T]
        threshold = self.args.pos_threshold if hasattr(self.args, "pos_threshold") else 0.5

        point_preds = (pos_probs >= threshold).detach().cpu().numpy().astype(np.int64)  # [B, T]
        label_np = label.detach().cpu().numpy().astype(np.int64)                         # [B, T]
        mask_np = mask.detach().cpu().numpy().astype(np.int64)                           # [B, T]

        fs = getattr(self.args, "fs", 256.0)
        merge_gap_sec = 0.2
        min_dur_sec = 0.5
        max_dur_sec = 3.0

        all_preds_list = []
        all_labels_list = []

        for b in range(B):
            seq_pred = point_preds[b]   # [T]
            seq_label = label_np[b]     # [T]
            seq_mask = mask_np[b]       # [T]

            events = self._postprocess_spindle_events(
                seq_pred,
                seq_mask,
                fs=fs,
                merge_gap_sec=merge_gap_sec,
                min_dur_sec=min_dur_sec,
                max_dur_sec=max_dur_sec,
            )

            seq_pred_post = self._events_to_point_preds(events, seq_mask)

            valid_idx = seq_mask > 0
            if np.any(valid_idx):
                all_preds_list.append(seq_pred_post[valid_idx])
                all_labels_list.append(seq_label[valid_idx])

        if all_preds_list:
            preds_valid_np = np.concatenate(all_preds_list, axis=0)
            labels_valid_np = np.concatenate(all_labels_list, axis=0)
        else:
            preds_valid_np = np.zeros(0, dtype=np.int64)
            labels_valid_np = np.zeros(0, dtype=np.int64)

        preds_valid = torch.from_numpy(preds_valid_np).long().to(outputs.device)
        labels_valid = torch.from_numpy(labels_valid_np).long().to(outputs.device)

        return loss, preds_valid, labels_valid

    def vali(self, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_trues = []
        event_tp = 0
        event_fp = 0
        event_fn = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)          # [B, T, 1]
                padding_mask = padding_mask.float().to(self.device)  # [B, T]
                label = label.to(self.device)                      # [B, T]

                outputs = self.model(batch_x, padding_mask, None, None)  # 期望 [B, T, C]

                loss, preds_flat, labels_flat = self._pointwise_loss_and_preds_postprocessed(
                    outputs, label, padding_mask, criterion
                )
                total_loss.append(loss.item())
                evt = self._event_level_metrics_from_batch(outputs, label, padding_mask)
                event_tp += evt["tp"]
                event_fp += evt["fp"]
                event_fn += evt["fn"]

                if preds_flat.numel() > 0:
                    all_preds.append(preds_flat.detach().cpu())
                    all_trues.append(labels_flat.detach().cpu())

        total_loss = np.average(total_loss) if total_loss else 0.0

        if all_preds:
            preds_cat = torch.cat(all_preds, dim=0).numpy()
            trues_cat = torch.cat(all_trues, dim=0).numpy()
            point_metrics = self._pointwise_metrics(preds_cat, trues_cat)
        else:
            point_metrics = dict(acc=0.0, precision=0.0, recall=0.0, f1=0.0)

        event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0.0
        event_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0.0
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0.0
        event_metrics = dict(tp=event_tp, fp=event_fp, fn=event_fn,
                             precision=event_precision, recall=event_recall, f1=event_f1)

        self.model.train()
        return total_loss, point_metrics, event_metrics

    def train(self, setting):
        # 直接使用预先划分好的 TRAIN / VAL 窗口
        train_data, train_loader = self._get_data(flag='TRAIN')
        train_loader = self._build_balanced_train_loader(train_data, train_loader)
        _, vali_loader = self._get_data(flag='VAL')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)            # [B, T, 1]
                padding_mask = padding_mask.float().to(self.device)  # [B, T]
                label = label.to(self.device)                        # [B, T]

                # 正窗口轻量 jitter 增强（只作用于包含正标签点的窗口）
                jitter_std = float(getattr(self.args, "pointseg_pos_jitter_std", 0.0))
                jitter_prob = float(getattr(self.args, "pointseg_pos_jitter_prob", 0.5))
                if jitter_std > 0:
                    with torch.no_grad():
                        pos_window_mask = (label.sum(dim=1) > 0)  # [B]
                        if pos_window_mask.any():
                            rand_mask = (torch.rand_like(pos_window_mask.float()) < jitter_prob) & pos_window_mask
                            if rand_mask.any():
                                noise = torch.randn_like(batch_x[rand_mask]) * jitter_std
                                batch_x[rand_mask] = batch_x[rand_mask] + noise

                outputs = self.model(batch_x, padding_mask, None, None)  # [B, T, C]

                # 只在第一个 epoch 的第一个 batch 打印一次形状，确认数据流是否正确
                if epoch == 0 and i == 0:
                    print("batch_x:", batch_x.shape)
                    print("label:", label.shape)
                    print("padding_mask:", padding_mask.shape)
                    print("outputs:", outputs.shape)

                loss, _, _ = self._pointwise_loss_and_preds(
                    outputs, label, padding_mask, criterion
                )
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) if train_loss else 0.0
            vali_loss, val_point_metrics, val_event_metrics = self.vali(vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} "
                "Vali Loss: {3:.3f} | Point F1: {4:.3f} | Event Prec: {5:.3f} "
                "Event Rec: {6:.3f} Event F1: {7:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_point_metrics["f1"],
                    val_event_metrics["precision"],
                    val_event_metrics["recall"],
                    val_event_metrics["f1"],
                )
            )
            # 默认基线：事件级 F1 作为核心早停指标
            early_stopping(-val_event_metrics["f1"], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        total_loss = []
        all_preds = []
        all_trues = []
        event_tp = 0
        event_fp = 0
        event_fn = 0

        self.model.eval()
        criterion = self._select_criterion()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                loss, preds_flat, labels_flat = self._pointwise_loss_and_preds_postprocessed(
                    outputs, label, padding_mask, criterion
                )
                total_loss.append(loss.item())
                evt = self._event_level_metrics_from_batch(outputs, label, padding_mask)
                event_tp += evt["tp"]
                event_fp += evt["fp"]
                event_fn += evt["fn"]

                if preds_flat.numel() > 0:
                    all_preds.append(preds_flat.detach().cpu())
                    all_trues.append(labels_flat.detach().cpu())

        total_loss = np.average(total_loss) if total_loss else 0.0

        if all_preds:
            preds_cat = torch.cat(all_preds, dim=0).numpy()
            trues_cat = torch.cat(all_trues, dim=0).numpy()
            point_metrics = self._pointwise_metrics(preds_cat, trues_cat)
        else:
            point_metrics = dict(acc=0.0, precision=0.0, recall=0.0, f1=0.0)

        event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0.0
        event_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0.0
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0.0
        event_metrics = dict(tp=event_tp, fp=event_fp, fn=event_fn,
                             precision=event_precision, recall=event_recall, f1=event_f1)

        print(
            "test point-wise metrics: "
            "Acc={acc:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}".format(**point_metrics)
        )
        print(
            "test event-level metrics (one-to-one overlap): "
            "TP={tp}, FP={fp}, FN={fn}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}".format(**event_metrics)
        )

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result_pointseg.txt", 'a')
        f.write(setting + "  \n")
        f.write('point_loss:{}'.format(total_loss))
        f.write('\n')
        f.write(
            'point_acc:{:.6f}, point_prec:{:.6f}, point_rec:{:.6f}, point_f1:{:.6f}'.format(
                point_metrics["acc"], point_metrics["precision"], point_metrics["recall"], point_metrics["f1"]
            )
        )
        f.write('\n')
        f.write(
            'event_tp:{}, event_fp:{}, event_fn:{}, event_prec:{:.6f}, event_rec:{:.6f}, event_f1:{:.6f}'.format(
                event_metrics["tp"], event_metrics["fp"], event_metrics["fn"],
                event_metrics["precision"], event_metrics["recall"], event_metrics["f1"]
            )
        )
        f.write('\n\n')
        f.close()
        return
