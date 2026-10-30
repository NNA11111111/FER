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
)
import numpy as np


class SoftWeightedKLDivLoss(nn.Module):
    """
    Kết hợp học từ phân phối 10 phiếu (Soft Label / KL-Divergence) với trọng số lớp
    dạng căn bậc hai nghịch đảo (Square-Root Class Weight) hoặc điều biến Focal (Soft Focal Loss)
    giúp xử lý triệt để mất cân bằng lớp cho Disgust và Contempt.
    """
    def __init__(self, class_weights=None, gamma=0.0):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.gamma = gamma

    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # KL(p || q) = sum_c p_c * (log(p_c) - log(q_c))
        safe_targets = torch.clamp(soft_targets, min=1e-8)
        kl_per_class = soft_targets * (torch.log(safe_targets) - log_probs)

        if self.gamma > 0.0:
            focal_factor = (1.0 - probs) ** self.gamma
            kl_per_class = focal_factor * kl_per_class

        if self.class_weights is not None:
            kl_per_class = kl_per_class * self.class_weights.unsqueeze(0)

        return kl_per_class.sum(dim=1).mean()


def compute_epoch_metrics(y_true, y_pred):
    """
    Tính bộ 3 chỉ số chuẩn cho từng Epoch (Train / Validation / Test):
    Accuracy, Macro F1-Score (chỉ số chính) và Weighted F1-Score.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, macro_f1, weighted_f1


def summarize_multi_seed_results(seed_metrics_list, model_name="CustomCNN"):
    """
    Tổng hợp và in báo cáo Trung bình ± Độ lệch chuẩn (Mean ± Std) qua nhiều lần chạy (multi-seed)
    để đưa trực tiếp vào báo cáo hoặc bài báo khoa học.
    """
    accs = [m["acc"] * 100 for m in seed_metrics_list]
    macro_f1s = [m["macro_f1"] * 100 for m in seed_metrics_list]
    weighted_f1s = [m["weighted_f1"] * 100 for m in seed_metrics_list]
    disgust_recalls = [m["disgust_recall"] * 100 for m in seed_metrics_list]
    contempt_recalls = [m["contempt_recall"] * 100 for m in seed_metrics_list]

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
        "disgust_recall_mean": np.mean(disgust_recalls),
        "disgust_recall_std": np.std(disgust_recalls, ddof=ddof),
        "contempt_recall_mean": np.mean(contempt_recalls),
        "contempt_recall_std": np.std(contempt_recalls, ddof=ddof),
    }

    print(f"\n==================================================================")
    print(f" TỔNG HỢP KẾT QUẢ {len(seed_metrics_list)} LẦN CHẠY (SEEDS) - MÔ HÌNH: {model_name}")
    print(f"==================================================================")
    print(f"  • Macro F1 (Chính) : {summary['macro_f1_mean']:.2f}% ± {summary['macro_f1_std']:.2f}%")
    print(f"  • Accuracy         : {summary['acc_mean']:.2f}% ± {summary['acc_std']:.2f}%")
    print(f"  • Weighted F1      : {summary['weighted_f1_mean']:.2f}% ± {summary['weighted_f1_std']:.2f}%")
    print(f"  • Recall Disgust   : {summary['disgust_recall_mean']:.2f}% ± {summary['disgust_recall_std']:.2f}%")
    print(f"  • Recall Contempt  : {summary['contempt_recall_mean']:.2f}% ± {summary['contempt_recall_std']:.2f}%")
    print(f"==================================================================\n")
    return summary


def plot_learning_curves(history, phase1_epochs=None):
    """
    Vẽ biểu đồ đường cong học tập (Learning Curves) qua các Epoch gồm 3 đồ thị:
    1. Đường cong hàm mất mát (Train Loss vs Validation Loss)
    2. Đường cong độ chính xác (Train Accuracy vs Validation Accuracy)
    3. Đường cong F1-Score (Macro F1 & Weighted F1 trên Train và Validation)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(18, 5))

    # 1. Biểu đồ Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu Fine-tune")
    plt.title("Đường cong Hàm mất mát (Loss Curve)")
    plt.xlabel("Epoch")
    plt.ylabel("KL-Divergence Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # 2. Biểu đồ Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [x * 100 for x in history["train_acc"]], label="Train Acc", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [x * 100 for x in history["val_acc"]], label="Val Acc", color="#d62728", linewidth=2)
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu Fine-tune")
    plt.title("Đường cong Độ chính xác (Accuracy Curve)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # 3. Biểu đồ Macro F1 & Weighted F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [x * 100 for x in history["train_macro_f1"]], label="Train Macro F1", color="#9467bd", linewidth=1.8, linestyle="--")
    plt.plot(epochs, [x * 100 for x in history["val_macro_f1"]], label="Val Macro F1 (Chính)", color="#8c564b", linewidth=2.2)
    plt.plot(epochs, [x * 100 for x in history["val_weighted_f1"]], label="Val Weighted F1", color="#e377c2", linewidth=1.8)
    if phase1_epochs is not None and phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs, color="red", linestyle=":", alpha=0.8, label="Bắt đầu Fine-tune")
    plt.title("Đường cong F1-Score (Macro & Weighted F1)")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score (%)")
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


def ten_crop_inference(model, image_tensor, pad_size=4, crop_size=48):
    """
    Kỹ thuật Ten-Crop Averaging chuẩn theo Pramerdorfer & Kampel (2016):
    Đệm viền (Pad=4, reflect) đưa ảnh từ 48x48 lên 56x56, sau đó cắt TenCrop(48).
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
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)
        predicted_label = torch.argmax(avg_probs).item()

    return predicted_label, avg_probs.cpu().numpy()


def generate_gradcam(model, image_tensor, target_class=None, target_layer=None):
    """
    Tính Grad-CAM (Selvaraju et al., 2017) tại tầng tích chập cuối cùng (6x6).
    Hỗ trợ tự động cả ResSEBlock (model.block4.conv2) và Sequential CustomCNN cũ (model.block4[3]).
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
    Xuất báo cáo kiểm định lấy Macro F1 làm chỉ số chính, kèm Recall của Disgust và Contempt.
    """
    labels = [emotion_dict[i] for i in range(len(emotion_dict))]
    acc, macro_f1, weighted_f1 = compute_epoch_metrics(y_true, y_pred)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    print(f"=== TỔNG HỢP CHỈ SỐ TRÊN TẬP {title_prefix.upper()} ===")
    print(
        f"Macro F1 (Chỉ số chính): {macro_f1*100:.2f}% | Accuracy: {acc*100:.2f}% | Weighted F1: {weighted_f1*100:.2f}%"
    )
    print(
        f"Recall lớp hiếm -> Disgust: {per_class_recall[5]*100:.2f}% | Contempt: {per_class_recall[7]*100:.2f}%\n"
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
        "disgust_recall": float(per_class_recall[5]),
        "contempt_recall": float(per_class_recall[7]),
    }