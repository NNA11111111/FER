import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# Import kiến trúc mạng CNN của bạn
from models import CustomCNN

# Cập nhật từ điển 8 lớp theo đúng chuẩn FER+ thống nhất toàn dự án
EMOTION_DICT = {
    0: 'Neutral', 
    1: 'Happiness', 
    2: 'Surprise', 
    3: 'Sadness', 
    4: 'Anger', 
    5: 'Disgust', 
    6: 'Fear', 
    7: 'Contempt'
}

def preprocess_live_frame(face_crop):
    """
    Áp dụng chính xác Pipeline tiền xử lý cho khung ảnh cắt từ Webcam.
    """
    # 1. Định hình kích thước 48x48
    face_resized = cv2.resize(face_crop, (48, 48))
    
    # 2. Đảm bảo ảnh xám 1 kênh
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized
        
    # 3. Cân bằng sáng cục bộ (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face_clahe = clahe.apply(face_gray)
    
    # 4. Chuẩn hóa Pixel về [-1.0, 1.0]
    X = face_clahe.astype(np.float32) / 255.0
    X_norm = (X - 0.5) / 0.5
    
    # Chuyển thành Tensor cho PyTorch với shape (Batch=1, Channel=1, H=48, W=48)
    tensor_img = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return tensor_img

def run_realtime_inference(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    
    # Khởi tạo mạng 8 lớp và nạp trọng số đã huấn luyện
    model = CustomCNN(num_classes=8)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval() # Tắt Dropout, cố định BatchNorm
        print("Tải trọng số thành công. Đang mở Webcam...")
    except Exception as e:
        print(f"Lỗi: Không tìm thấy hoặc không thể đọc file '{model_path}'.\nChi tiết: {e}")
        return

    # Khởi tạo Face Detector của OpenCV (HaarCascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Mở Webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Lật ngược ảnh để tránh hiệu ứng gương
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
        
        for (x, y, w, h) in faces:
            # 2. Cắt ô khuôn mặt (Bounding Box)
            face_crop = gray_frame[y:y+h, x:x+w]
            
            # 3. Tiền xử lý
            tensor_input = preprocess_live_frame(face_crop).to(device)
            
            # 4. Dự đoán cảm xúc
            with torch.no_grad():
                outputs = model(tensor_input)
                pred_idx = torch.argmax(outputs, dim=1).item()
                emotion_label = EMOTION_DICT[pred_idx]
            
            # 5. Vẽ khung vuông và in nhãn cảm xúc lên màn hình
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        # Hiển thị Video
        cv2.imshow("Real-time FER+", frame)
        
        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()