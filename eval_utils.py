import copy
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)
import numpy as np


def compute_smooth_class_weights(class_counts, power=0.25, max_ratio=3.0):
    """
    Tính trọng số lớp êm dịu theo luỹ thừa nghịch đảo (Smooth Inverse-Power Class Weights):
        w_c = (1 / n_c)^power, giới hạn tỷ lệ max(w)/min(w) <= max_ratio.
    Khắc phục triệt để lỗi trọng số quá gắt (gấp 12.3 lần ở phiên bản cũ với beta=0.997)
    vốn làm sụp đổ Precision của Disgust và Contempt ở Giai đoạn 2.
    Với power=0.25 trên FER+, trọng số của Disgust/Contempt chỉ gấp ~2.7 lần Neutral,
    giúp tăng Recall lớp hiếm mà vẫn giữ Precision cao.
    """
    counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    raw_weights = np.power(1.0 / counts, power)
    min_w = raw_weights.min()
    clipped_weights = np.minimum(raw_weights, min_w * float(max_ratio))
    norm_weights = clipped_weights / clipped_weights.mean()
    return norm_weights.astype(np.float32)


def compute_effective_number_weights(class_counts, beta=0.99, max_ratio=3.2):
    """
    Tính trọng số lớp theo Số lượng Mẫu Hiệu dụng (Cui et al., CVPR 2019)
    CÓ GIỚI HẠN TỶ LỆ TỐI ĐA (max_ratio=3.2) để ngăn chặn hiện tượng bùng nổ False Positives
    trên các lớp cực hiếm (Disgust, Contempt).
    """
    counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    min_w = weights.min()
    weights = np.minimum(weights, min_w * float(max_ratio))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_class_priors(class_counts):
    """
    Tính phân phối xác suất tiên nghiệm (Class Priors pi_c)
    phục vụ Hiệu chỉnh Logit (Logit Adjustment - Menon et al., ICLR 2021).
    """
    counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    priors = counts / counts.sum()
    return priors.astype(np.float32)


def get_cosine_gamma(epoch, total_epochs, warmup_epochs=12, max_gamma=0.35):
    """
    Chiến lược Điều phối Gamma Tự động ở mức toàn mẫu (Sample-level Focal Gamma Annealing):
    - Giữ gamma = 0.0 trong `warmup_epochs` đầu tiên.
    - Sau đó tăng nhẹ theo hàm Cosine lên `max_gamma` (mặc định 0.35 thay vì 1.0)
      để tập trung vào các mẫu khó mà không làm biến dạng phân phối nhãn mềm.
    """
    if max_gamma <= 0.0 or epoch < warmup_epochs:
        return 0.0
    progress = float(epoch - warmup_epochs + 1) / float(max(1, total_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(max_gamma * cosine_scale)


class SoftWeightedKLDivLoss(nn.Module):
    r"""
    Hàm mất mát Sample-Level Focal KL-Divergence trên nhãn mềm 10 phiếu (FER+) đã sửa lỗi lệch xác suất:
    1. Khắc phục lỗi ở phiên bản cũ: Phiên bản cũ nhân `(1 - probs)^gamma` riêng lẻ vào từng lớp `c`,
       làm triệt tiêu đạo hàm của cảm xúc chính và phóng đại nhiễu của cảm xúc phụ khi gamma tăng cao sau Epoch 30.
    2. Phiên bản mới tính hệ số điều biến Focal ở MỨC TOÀN MẪU (Sample-Level Alignment):
           p_align = \sum_{c=1}^C y_c \cdot p_c
           focal_factor = (1 - p_align)^\gamma
       Nhờ nhân cùng một hệ số `focal_factor` cho cả vector nhãn mềm của mẫu `i`, tỷ lệ xác suất mục tiêu
       giữa 8 lớp được bảo toàn tuyệt đối (strictly proper scoring rule), giúp mạng học càng lâu càng chính xác!
    3. Hỗ trợ Logit Adjustment trong lúc huấn luyện (`logit_adj_tau`) và Trọng số mẫu đám đông (`sample_weights`).
    """
    def __init__(self, class_weights=None, class_priors=None, logit_adj_tau=0.0, gamma=0.0):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        has_tau = False
        if logit_adj_tau is not None:
            if np.ndim(logit_adj_tau) == 0:
                has_tau = float(logit_adj_tau) > 0.0
            else:
                has_tau = np.any(np.asarray(logit_adj_tau, dtype=np.float32) != 0.0)

        if class_priors is not None and has_tau:
            log_priors = torch.log(torch.clamp(torch.tensor(class_priors, dtype=torch.float32), min=1e-12))
            tau_tensor = torch.as_tensor(logit_adj_tau, dtype=torch.float32)
            self.register_buffer("logit_adjustment", tau_tensor * log_priors)
        else:
            self.logit_adjustment = None

        self.gamma = float(gamma)

    def set_gamma(self, new_gamma):
        """Cập nhật hệ số Focal gamma động theo từng epoch."""
        self.gamma = float(max(0.0, new_gamma))

    def forward(self, logits, soft_targets, sample_weights=None):
        adjusted_logits = (
            logits + self.logit_adjustment.unsqueeze(0)
            if self.logit_adjustment is not None
            else logits
        )

        log_probs = F.log_softmax(adjusted_logits, dim=1)
        probs = torch.exp(log_probs)

        safe_targets = torch.clamp(soft_targets, min=1e-8)
        kl_per_class = torch.where(
            soft_targets > 0.0,
            soft_targets * (torch.log(safe_targets) - log_probs),
            torch.zeros_like(log_probs),
        )

        if self.class_weights is not None:
            kl_per_class = kl_per_class * self.class_weights.unsqueeze(0)

        loss_per_sample = kl_per_class.sum(dim=1)

        # Điều biến Focal ở mức toàn mẫu (giữ nguyên tính chuẩn xác suất của KL-Divergence)
        if self.gamma > 0.0:
            p_align = (soft_targets * probs).sum(dim=1)
            one_minus_p = torch.clamp(1.0 - p_align, min=1e-6, max=1.0)
            focal_factor = one_minus_p ** self.gamma
            loss_per_sample = focal_factor * loss_per_sample

        if sample_weights is not None:
            w = sample_weights.to(loss_per_sample.device, dtype=loss_per_sample.dtype).view(-1)
            return (loss_per_sample * w).sum() / torch.clamp(w.sum(), min=1e-8)

        return loss_per_sample.mean()


def _is_zero_tau(tau):
    if tau is None:
        return True
    if isinstance(tau, (int, float, np.number)):
        return float(tau) == 0.0
    if isinstance(tau, torch.Tensor):
        return bool(torch.all(tau == 0.0).item())
    arr = np.asarray(tau, dtype=np.float32)
    return bool(np.all(arr == 0.0))


def apply_posthoc_logit_adjustment(logits, class_priors, tau=0.10):
    r"""
    Hiệu chỉnh Logit Hậu nghiệm (Post-hoc Logit Adjustment - Menon et al., ICLR 2021):
        \tilde{z}_c = z_c - \tau_c \ln \pi_c
    Lưu ý: Nếu trong lúc train đã dùng `stage1_tau` (ví dụ 0.30), thì `posthoc_tau` chỉ cần đặt nhỏ
    (ví dụ 0.10) để tránh cộng chồng làm bùng nổ False Positives trên các lớp hiếm.
    """
    if _is_zero_tau(tau) or class_priors is None:
        return logits
    priors_tensor = torch.as_tensor(class_priors, dtype=logits.dtype, device=logits.device)
    log_priors = torch.log(torch.clamp(priors_tensor, min=1e-12))
    tau_tensor = torch.as_tensor(tau, dtype=logits.dtype, device=logits.device)
    if tau_tensor.ndim == 0:
        return logits - tau_tensor * log_priors.unsqueeze(0)
    return logits - (tau_tensor * log_priors).unsqueeze(0)


def optimize_classwise_posthoc_tau(
    val_logits,
    val_targets,
    class_priors,
    base_tau=0.10,
    max_acc_drop=0.004,
    max_rare_recall_drop=0.03,
    alpha_composite=0.5,
    rare_classes=(5, 6, 7),
    num_rounds=2,
    l2_reg=0.06,
):
    """
    Tối ưu hoá ngưỡng hiệu chỉnh hậu nghiệm CHỐNG OVERFIT TRÊN TẬP VALIDATION NHỎ
    VÀ BẢO VỆ CẢ PRECISION LẪN RECALL (F1) CHO CẢ 3 LỚP HIẾM (Disgust=5, Fear=6, Contempt=7):
    - Khắc phục lỗi ở phiên bản cũ: Phiên bản cũ cho phép tau_c chạy rộng từ [0.20, 0.70] trên tập Val
      chỉ có 28 mẫu Disgust và 30 mẫu Contempt, dẫn tới việc đẩy tau_0 (Neutral) lên 0.70 và gây ra
      80-110 dự đoán nhầm (False Positives) sang Contempt/Disgust trên tập Test!
    - Giải pháp mới:
      1. Bước 1: Quét tìm hệ số vô hướng `best_scalar` chung cho cả 8 lớp trong miền [0.0, 0.30]
         (vì mô hình đã học sẵn `stage1_tau=0.30` trong lúc train).
      2. Bước 2: Chỉ cho phép từng lớp `tau_c` tinh chỉnh trong biên độ rất hẹp `[best_scalar - 0.08, best_scalar + 0.08]`
         với hệ số phạt L2 mạnh (`l2_reg=0.06`) và tối ưu trực tiếp F1-score của cả 3 lớp hiếm `(5, 6, 7)`.
    """
    if isinstance(val_logits, torch.Tensor):
        logits_np = val_logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(val_logits, dtype=np.float32)

    targets_np = np.asarray(val_targets, dtype=np.int64)
    priors_np = np.maximum(np.asarray(class_priors, dtype=np.float32), 1e-12)
    log_priors_np = np.log(priors_np)
    num_classes = logits_np.shape[1]

    def eval_tau_vec(tau_vec, ref_tau):
        adj = logits_np - (tau_vec * log_priors_np)[np.newaxis, :]
        preds = np.argmax(adj, axis=1)
        acc = accuracy_score(targets_np, preds)
        macro_f1 = f1_score(targets_np, preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(targets_np, preds, average=None, zero_division=0)
        per_class_rec = recall_score(targets_np, preds, average=None, zero_division=0)
        comp = alpha_composite * acc + (1.0 - alpha_composite) * macro_f1

        rare_f1_mean = float(np.mean([per_class_f1[c] for c in rare_classes]))
        l2_pen = float(np.mean((tau_vec - ref_tau) ** 2))

        # Hàm mục tiêu kết hợp Val Composite + thưởng F1 của 3 lớp hiếm (Disgust, Fear, Contempt) - phạt L2 mạnh
        obj = comp + 0.12 * rare_f1_mean - float(l2_reg) * l2_pen
        return obj, acc, macro_f1, comp, per_class_rec, per_class_f1

    # Bước 1: Tìm hằng số vô hướng scalar_tau tốt nhất trên miền [0.0, 0.30]
    best_scalar = float(base_tau)
    best_scalar_obj = -1e9
    for cand_s in np.linspace(0.0, 0.30, 31, dtype=np.float32):
        vec_s = np.full(num_classes, cand_s, dtype=np.float32)
        obj_s, _, _, _, _, _ = eval_tau_vec(vec_s, cand_s)
        if obj_s > best_scalar_obj:
            best_scalar_obj = obj_s
            best_scalar = float(cand_s)

    base_tau_vec = np.full(num_classes, best_scalar, dtype=np.float32)
    best_tau = base_tau_vec.copy()
    best_obj, base_acc, base_f1, base_comp, base_rec, base_pf1 = eval_tau_vec(best_tau, best_scalar)
    min_allowed_acc = base_acc - float(max_acc_drop)

    # Bước 2: Tinh chỉnh nhẹ từng lớp trong khoảng hẹp ±0.08 quanh best_scalar để chống Overfit
    search_order = list(rare_classes) + [c for c in range(num_classes) if c not in rare_classes]
    low_bound = max(0.0, best_scalar - 0.08)
    high_bound = min(0.35, best_scalar + 0.08)
    candidate_grid = np.linspace(low_bound, high_bound, 9, dtype=np.float32)

    for _ in range(num_rounds):
        improved = False
        for c in search_order:
            for cand in candidate_grid:
                trial_tau = best_tau.copy()
                trial_tau[c] = cand
                obj, acc, _, _, rec, pf1 = eval_tau_vec(trial_tau, best_scalar)

                rare_ok = all(
                    rec[r] >= base_rec[r] - max_rare_recall_drop and pf1[r] >= base_pf1[r] - 0.02
                    for r in rare_classes
                )

                if acc >= min_allowed_acc and rare_ok and obj > best_obj + 1e-5:
                    best_obj = obj
                    best_tau = trial_tau
                    improved = True
        if not improved:
            break

    _, final_acc, final_f1, final_comp, final_rec, final_pf1 = eval_tau_vec(best_tau, best_scalar)
    print("\n=== TỐI ƯU HIỆU CHỈNH LOGIT HẬU NGHIỆM CHỐNG OVERFIT (3 LỚP HIẾM: DISGUST, FEAR, CONTEMPT) ===")
    print(f"  • Scalar tau tối ưu ({best_scalar:.3f}) -> Val Comp: {base_comp*100:.2f}% | Val Macro-F1: {base_f1*100:.2f}% | "
          f"Val Acc: {base_acc*100:.2f}%")
    print(f"  • Vector tau tinh chỉnh -> Val Comp: {final_comp*100:.2f}% | Val Macro-F1: {final_f1*100:.2f}% | "
          f"Val Acc: {final_acc*100:.2f}%")
    print(f"  • F1 trên Val (3 lớp hiếm) -> Disgust: {final_pf1[5]*100:.2f}% (Rec {final_rec[5]*100:.1f}%) | "
          f"Fear: {final_pf1[6]*100:.2f}% (Rec {final_rec[6]*100:.1f}%) | "
          f"Contempt: {final_pf1[7]*100:.2f}% (Rec {final_rec[7]*100:.1f}%)")
    print(f"  • Vector tau_c (8 lớp): {np.round(best_tau, 3).tolist()}")
    return best_tau


class ModelEMA:
    """
    Trung bình trượt hàm mũ trọng số mô hình (Exponential Moving Average - Polyak & Juditsky, 1992).
    """
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)


def compute_epoch_metrics(y_true, y_pred, alpha_composite=0.5):
    """
    Tính bộ chỉ số chuẩn cho từng Epoch (Train / Validation / Test):
    - Accuracy, Macro F1-Score, Weighted F1-Score, Composite Score = alpha * Acc + (1 - alpha) * Macro_F1
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    composite_score = alpha_composite * acc + (1.0 - alpha_composite) * macro_f1
    return acc, macro_f1, weighted_f1, composite_score


def summarize_multi_seed_results(seed_metrics_list, model_name="CustomCNN"):
    """
    Tổng hợp và in báo cáo Trung bình ± Độ lệch chuẩn (Mean ± Std) qua nhiều lần chạy (multi-seed).
    """
    accs = [m["acc"] * 100 for m in seed_metrics_list]
    macro_f1s = [m["macro_f1"] * 100 for m in seed_metrics_list]
    weighted_f1s = [m["weighted_f1"] * 100 for m in seed_metrics_list]
    composites = [m.get("composite", 0.5 * m["acc"] + 0.5 * m["macro_f1"]) * 100 for m in seed_metrics_list]
    disgust_recalls = [m["disgust_recall"] * 100 for m in seed_metrics_list]
    contempt_recalls = [m["contempt_recall"] * 100 for m in seed_metrics_list]
    fear_recalls = [m.get("fear_recall", 0.0) * 100 for m in seed_metrics_list]
    disgust_f1s = [m.get("disgust_f1", 0.0) * 100 for m in seed_metrics_list]
    contempt_f1s = [m.get("contempt_f1", 0.0) * 100 for m in seed_metrics_list]
    fear_f1s = [m.get("fear_f1", 0.0) * 100 for m in seed_metrics_list]

    ddof = 1 if len(seed_metrics_list) > 1 else 0

    summary = {
        "model": model_name,
        "num_seeds": len(seed_metrics_list),
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs, ddof=ddof),
        "macro_f1_mean": np.mean(macro_f1s),
        "macro_f1_std": np.std(macro_f1s, ddof=ddof),
        "weighted_f1_mean": np.mean(weighted_f1s),
        "weighted_f1_std": np.std(weighted_f1s, ddof=ddof),
        "composite_mean": np.mean(composites),
        "composite_std": np.std(composites, ddof=ddof),
        "disgust_recall_mean": np.mean(disgust_recalls),
        "disgust_recall_std": np.std(disgust_recalls, ddof=ddof),
        "contempt_recall_mean": np.mean(contempt_recalls),
        "contempt_recall_std": np.std(contempt_recalls, ddof=ddof),
        "fear_recall_mean": np.mean(fear_recalls),
        "fear_recall_std": np.std(fear_recalls, ddof=ddof),
        "disgust_f1_mean": np.mean(disgust_f1s),
        "disgust_f1_std": np.std(disgust_f1s, ddof=ddof),
        "contempt_f1_mean": np.mean(contempt_f1s),
        "contempt_f1_std": np.std(contempt_f1s, ddof=ddof),
        "fear_f1_mean": np.mean(fear_f1s),
        "fear_f1_std": np.std(fear_f1s, ddof=ddof),
    }

    print(f"\n==================================================================")
    print(f" TỔNG HỢP KẾT QUẢ {len(seed_metrics_list)} LẦN CHẠY (SEEDS) - MÔ HÌNH: {model_name}")
    print(f"==================================================================")
    print(f"  • Macro F1 (Chính)      : {summary['macro_f1_mean']:.2f}% ± {summary['macro_f1_std']:.2f}%")
    print(f"  • Composite (0.5A+0.5F) : {summary['composite_mean']:.2f}% ± {summary['composite_std']:.2f}%")
    print(f"  • Accuracy              : {summary['acc_mean']:.2f}% ± {summary['acc_std']:.2f}%")
    print(f"  • Weighted F1           : {summary['weighted_f1_mean']:.2f}% ± {summary['weighted_f1_std']:.2f}%")
    print(f"  • Disgust  (F1 / Rec)   : F1={summary['disgust_f1_mean']:.2f}%±{summary['disgust_f1_std']:.2f}% | Rec={summary['disgust_recall_mean']:.2f}%±{summary['disgust_recall_std']:.2f}%")
    print(f"  • Fear     (F1 / Rec)   : F1={summary['fear_f1_mean']:.2f}%±{summary['fear_f1_std']:.2f}% | Rec={summary['fear_recall_mean']:.2f}%±{summary['fear_recall_std']:.2f}%")
    print(f"  • Contempt (F1 / Rec)   : F1={summary['contempt_f1_mean']:.2f}%±{summary['contempt_f1_std']:.2f}% | Rec={summary['contempt_recall_mean']:.2f}%±{summary['contempt_recall_std']:.2f}%")
    print(f"==================================================================\n")
    return summary


def plot_learning_curves(history, phase1_epochs=None):
    """
    Vẽ biểu đồ đường cong học tập (Learning Curves) qua các Epoch gồm 3 đồ thị:
    1. Đường cong hàm mất mát (Train Loss vs Validation Loss)
    2. Đường cong độ chính xác (Train Accuracy vs Validation Accuracy)
    3. Đường cong F1-Score & Composite Score trên Train và Validation
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu GĐ 2 (Block4+Head FT)")
    plt.title("Đường cong Hàm mất mát (Loss Curve)")
    plt.xlabel("Epoch")
    plt.ylabel("Sample-Focal KL-Divergence Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, [x * 100 for x in history["train_acc"]], label="Train Acc", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [x * 100 for x in history["val_acc"]], label="Val Acc", color="#d62728", linewidth=2)
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu GĐ 2 (Block4+Head FT)")
    plt.title("Đường cong Độ chính xác (Accuracy Curve)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, [x * 100 for x in history["train_macro_f1"]], label="Train Macro F1", color="#9467bd", linewidth=1.6, linestyle="--")
    plt.plot(epochs, [x * 100 for x in history["val_macro_f1"]], label="Val Macro F1", color="#8c564b", linewidth=2.2)
    if "val_composite" in history and len(history["val_composite"]) == len(epochs):
        plt.plot(epochs, [x * 100 for x in history["val_composite"]], label="Val Composite (0.5Acc+0.5F1)", color="#17becf", linewidth=2.0)
    plt.plot(epochs, [x * 100 for x in history["val_weighted_f1"]], label="Val Weighted F1", color="#e377c2", linewidth=1.6)
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu GĐ 2 (Block4+Head FT)")
    plt.title("Đường cong F1-Score & Composite")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def denormalize_image(img_tensor):
    """
    Đảo ngược chuẩn hóa từ miền [-1.0, 1.0] về lại [0.0, 1.0] để hiển thị đúng độ sáng gốc.
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().squeeze().numpy()
    else:
        img = np.squeeze(img_tensor)
    img = img * 0.5 + 0.5
    return np.clip(img, 0.0, 1.0)


def ten_crop_inference(model, image_tensor, pad_size=4, crop_size=48, class_priors=None, posthoc_tau=0.0):
    """
    Kỹ thuật Ten-Crop Averaging chuẩn theo Pramerdorfer & Kampel (2016)
    kết hợp tùy chọn Hiệu chỉnh Logit Hậu nghiệm.
    """
    device = next(model.parameters()).device

    ten_crop_transform = transforms.Compose([
        transforms.Pad(padding=pad_size, padding_mode="reflect"),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
    ])

    crops = ten_crop_transform(image_tensor).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(crops)
        if class_priors is not None and not _is_zero_tau(posthoc_tau):
            logits = apply_posthoc_logit_adjustment(logits, class_priors, tau=posthoc_tau)
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)
        predicted_label = torch.argmax(avg_probs).item()

    return predicted_label, avg_probs.cpu().numpy()


def generate_gradcam(model, image_tensor, target_class=None, target_layer=None):
    """
    Tính Grad-CAM (Selvaraju et al., 2017) tại tầng tích chập cuối cùng (6x6).
    """
    device = next(model.parameters()).device
    model.eval()

    if target_layer is None:
        if hasattr(model, "block4"):
            if hasattr(model.block4, "conv2"):
                target_layer = model.block4.conv2
            elif isinstance(model.block4, nn.Sequential):
                target_layer = model.block4[3]
            else:
                target_layer = model.block4
        else:
            raise ValueError("Cần chỉ định target_layer cho kiến trúc mạng này.")

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    if len(image_tensor.shape) == 3:
        inp_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        inp_tensor = image_tensor.to(device)

    model.zero_grad()
    logits = model(inp_tensor)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[0, target_class]
    score.backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients[0][0]
    fmaps = activations[0][0]
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(fmaps.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = F.relu(cam).cpu().numpy()
    cam = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_CUBIC)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam, target_class


def generate_saliency_map(model, image_tensor, target_class=None):
    return generate_gradcam(model, image_tensor, target_class=target_class)


def evaluate_private_test(y_true, y_pred, emotion_dict, title_prefix="Test"):
    """
    Xuất báo cáo kiểm định lấy Macro F1 & Composite Score làm chỉ số chính,
    kèm F1 & Recall của cả 3 lớp hiếm: Disgust, Fear và Contempt.
    """
    labels = [emotion_dict[i] for i in range(len(emotion_dict))]
    acc, macro_f1, weighted_f1, composite = compute_epoch_metrics(y_true, y_pred)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f"=== TỔNG HỢP CHỈ SỐ TRÊN TẬP {title_prefix.upper()} ===")
    print(
        f"Macro F1: {macro_f1*100:.2f}% | Composite (0.5Acc+0.5F1): {composite*100:.2f}% | "
        f"Accuracy: {acc*100:.2f}% | Weighted F1: {weighted_f1*100:.2f}%"
    )
    print(
        f"Chi tiết 3 lớp hiếm -> Disgust (F1: {per_class_f1[5]*100:.2f}%, Rec: {per_class_recall[5]*100:.2f}%) | "
        f"Fear (F1: {per_class_f1[6]*100:.2f}%, Rec: {per_class_recall[6]*100:.2f}%) | "
        f"Contempt (F1: {per_class_f1[7]*100:.2f}%, Rec: {per_class_recall[7]*100:.2f}%)\n"
    )
    print("=== BÁO CÁO KIỂM ĐỊNH CHI TIẾT (CLASSIFICATION REPORT) ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Ma Trận Nhầm Lẫn (Confusion Matrix) - {title_prefix}")
    plt.ylabel("Nhãn Thực Tế (True Label)")
    plt.xlabel("Nhãn Dự Đoán (Predicted Label)")
    plt.show()

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "composite": composite,
        "disgust_recall": float(per_class_recall[5]),
        "fear_recall": float(per_class_recall[6]),
        "contempt_recall": float(per_class_recall[7]),
        "disgust_f1": float(per_class_f1[5]),
        "fear_f1": float(per_class_f1[6]),
        "contempt_f1": float(per_class_f1[7]),
    }