import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import numpy as np


def compute_epoch_metrics(y_true, y_pred):
    """
    Tính bộ 3 chỉ số chuẩn cho từng Epoch (Train / Validation / Test):
    Accuracy, Macro F1-Score và Weighted F1-Score.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, macro_f1, weighted_f1


def plot_learning_curves(history, phase1_epochs=None):
    """
    Vẽ biểu đồ đường cong học tập (Learning Curves) qua các Epoch gồm 3 đồ thị:
    1. Đường cong hàm mất mát (Train Loss vs Validation Loss)
    2. Đường cong độ chính xác (Train Accuracy vs Validation Accuracy)
    3. Đường cong F1-Score (Macro F1 & Weighted F1 trên Train và Validation)
    Nếu truyền vào phase1_epochs, hàm sẽ vẽ thêm vạch đứt đánh dấu ranh giới chuyển sang Fine-tuning.
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
    plt.plot(epochs, [x * 100 for x in history["val_macro_f1"]], label="Val Macro F1", color="#8c564b", linewidth=2.2)
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
    Đệm viền (Pad=4, reflect) đưa ảnh từ 48x48 lên 56x56, sau đó cắt TenCrop(48)
    để giữ nguyên 100% tỷ lệ khuôn mặt và độ sắc nét của pixel (không dùng Resize).
    """
    device = next(model.parameters()).device

    ten_crop_transform = transforms.Compose([
        transforms.Pad(padding=pad_size, padding_mode='reflect'),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    ])

    # Cắt và đưa lên thiết bị (GPU/CPU), chuyển thành shape: (10, 1, 48, 48)
    crops = ten_crop_transform(image_tensor).to(device)

    model.eval()
    with torch.no_grad():
        # Truyền 10 crop qua mô hình để lấy 10 vector xác suất
        logits = model(crops)
        probs = F.softmax(logits, dim=1)

        # Tính trung bình cộng theo chiều dọc (Mean of 10 inferences)
        avg_probs = probs.mean(dim=0)
        predicted_label = torch.argmax(avg_probs).item()

    return predicted_label, avg_probs.cpu().numpy()


def generate_gradcam(model, image_tensor, target_class=None, target_layer=None):
    """
    Tính Grad-CAM (Selvaraju et al., 2017) tại tầng tích chập cuối cùng (model.block4)
    để tạo bản đồ nhiệt mượt và tập trung chính xác vào vùng mắt, mũi, khóe miệng.
    """
    device = next(model.parameters()).device
    model.eval()

    if target_layer is None:
        # Mặc định móc vào lớp Conv2d cuối cùng trong block4 của CustomCNN
        target_layer = model.block4[3]

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

    # Tính trọng số kênh (Global Average Pooling trên chiều không gian của Gradient)
    grads = gradients[0][0]          # shape: (C, H', W') -> (512, 6, 6)
    fmaps = activations[0][0]        # shape: (C, H', W') -> (512, 6, 6)
    weights = grads.mean(dim=(1, 2)) # shape: (C,)

    # Tổ hợp tuyến tính các bản đồ đặc trưng và qua hàm ReLU
    cam = torch.zeros(fmaps.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = F.relu(cam).cpu().numpy()

    # Nội suy (Resize) bản đồ nhiệt từ (6, 6) lên kích thước ảnh gốc (48, 48)
    cam = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_CUBIC)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam, target_class


def generate_saliency_map(model, image_tensor, target_class=None):
    """
    Hàm giữ lại để tương thích ngược: gọi sang generate_gradcam để trả về bản đồ nhiệt mượt.
    """
    return generate_gradcam(model, image_tensor, target_class=target_class)


def evaluate_private_test(y_true, y_pred, emotion_dict):
    """
    Xuất báo cáo kiểm định: Accuracy, Macro F1, Weighted F1, Classification Report và Confusion Matrix.
    """
    labels = [emotion_dict[i] for i in range(len(emotion_dict))]
    acc, macro_f1, weighted_f1 = compute_epoch_metrics(y_true, y_pred)

    print("=== TỔNG HỢP CHỈ SỐ TRÊN TẬP PRIVATETEST ===")
    print(f"Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | Weighted F1: {weighted_f1*100:.2f}%\n")
    print("=== BÁO CÁO KIỂM ĐỊNH CHI TIẾT (CLASSIFICATION REPORT) ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix) trên PrivateTest')
    plt.ylabel('Nhãn Thực Tế (True Label)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    plt.show()