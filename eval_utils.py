import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def ten_crop_inference(model, image_tensor, crop_size=42, target_size=48):
    """
    Kỹ thuật Ten-Crop Averaging để tăng độ tin cậy khi đánh giá trên tập PrivateTest.
    Đầu vào: 
        - image_tensor: Tensor ảnh kích thước (1, 48, 48) hoặc (48, 48)
    Đầu ra:
        - predicted_label: Nhãn cảm xúc dự đoán
        - avg_probs: Vector xác suất (7,) sau khi trung bình cộng
    """
    device = next(model.parameters()).device
    
    # TenCrop trả về 10 crops (4 góc, 1 tâm + 5 bản lật ngang).
    # Resize các crop về lại target_size (48x48) để khớp với đầu vào của mạng CNN.
    ten_crop_transform = transforms.Compose([
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Resize((target_size, target_size))(crop) for crop in crops
        ]))
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


def generate_saliency_map(model, image_tensor, target_class=None):
    """
    Tính Saliency Map để trực quan hóa vùng ảnh mà CNN chú ý.
    """
    device = next(model.parameters()).device
    
    # Đảm bảo có batch dimension: (1, 1, 48, 48)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    image_tensor = image_tensor.to(device)
    
    # Bật requires_grad_() để theo dõi đạo hàm của tensor đầu vào
    image_tensor.requires_grad_()
    
    model.eval()
    logits = model(image_tensor)
    
    # Nếu không cung cấp target_class, lấy nhãn được dự đoán cao nhất
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
        
    # Lấy giá trị (score) của lớp mục tiêu trước khi qua Softmax
    score = logits[0, target_class]
    
    # Tính toán ngược (Backward) để lấy đạo hàm
    model.zero_grad()
    score.backward()
    
    # Saliency Map được tính bằng trị tuyệt đối của gradient đối với ma trận ảnh gốc
    saliency_map = image_tensor.grad.abs().squeeze().cpu().numpy()
    
    return saliency_map, target_class


def evaluate_private_test(y_true, y_pred, emotion_dict):
    """
    Xuất báo cáo kiểm định: Precision, Recall, F1-Score và vẽ Confusion Matrix.
    """
    labels = [emotion_dict[i] for i in range(len(emotion_dict))]
    
    print("=== BÁO CÁO KIỂM ĐỊNH (CLASSIFICATION REPORT) ===")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix) trên PrivateTest')
    plt.ylabel('Nhãn Thực Tế (True Label)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    plt.show()