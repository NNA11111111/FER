import time
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import CustomCNN

EMOTION_DICT = {
    0: "Neutral",
    1: "Happiness",
    2: "Surprise",
    3: "Sadness",
    4: "Anger",
    5: "Disgust",
    6: "Fear",
    7: "Contempt",
}


def preprocess_live_frame(face_crop):
    """
    Áp dụng chính xác Pipeline tiền xử lý cho khung ảnh cắt từ Webcam
    theo đúng thứ tự trong data_utils.py: Grayscale -> Resize(48, 48) -> CLAHE -> Norm [-1.0, 1.0].
    """
    if len(face_crop.shape) == 3:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_crop

    face_resized = cv2.resize(face_gray, (48, 48))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face_clahe = clahe.apply(face_resized)

    X = face_clahe.astype(np.float32) / 255.0
    X_norm = (X - 0.5) / 0.5

    tensor_img = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor_img


class MultiFaceTracker:
    """
    Quản lý hàng đợi xác suất (deque) riêng biệt cho từng khuôn mặt dựa trên khoảng cách tâm (Centroid)
    hoặc chỉ số IoU giữa các frame liên tiếp, giúp dự đoán ổn định khi có nhiều người trong khung hình.
    """
    def __init__(self, maxlen=5, max_distance=80.0, max_missed_frames=10):
        self.maxlen = maxlen
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.tracks = {}  # track_id -> {"centroid": (cx, cy), "bbox": (x,y,w,h), "queue": deque, "missed": int}
        self.next_id = 1

    def update(self, bboxes, probs_list):
        # Tăng số frame bị mất dấu của tất cả các track hiện có
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missed"] += 1

        assigned_track_ids = []
        used_tracks = set()

        for bbox, probs in zip(bboxes, probs_list):
            x, y, w, h = bbox
            cx, cy = x + w / 2.0, y + h / 2.0

            best_tid = None
            best_dist = self.max_distance

            for tid, tdata in self.tracks.items():
                if tid in used_tracks:
                    continue
                tcx, tcy = tdata["centroid"]
                dist = np.hypot(cx - tcx, cy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1
                self.tracks[best_tid] = {
                    "centroid": (cx, cy),
                    "bbox": bbox,
                    "queue": deque(maxlen=self.maxlen),
                    "missed": 0,
                }
            else:
                self.tracks[best_tid]["centroid"] = (cx, cy)
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["missed"] = 0

            self.tracks[best_tid]["queue"].append(probs)
            used_tracks.add(best_tid)
            assigned_track_ids.append(best_tid)

        # Xoá các track đã rời khỏi khung hình quá max_missed_frames
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed_frames:
                del self.tracks[tid]

        return assigned_track_ids

    def get_smoothed_prediction(self, track_id):
        avg_probs = np.mean(self.tracks[track_id]["queue"], axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx]) * 100.0
        return pred_idx, confidence


def run_realtime_inference(model_path="best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    model = CustomCNN(num_classes=8)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("Tải trọng số thành công. Đang mở Webcam...")
    except Exception as e:
        print(f"Lỗi: Không tìm thấy hoặc không thể đọc file '{model_path}'.\nChi tiết: {e}")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    tracker = MultiFaceTracker(maxlen=5, max_distance=80.0, max_missed_frames=10)

    cap = cv2.VideoCapture(0)
    prev_time = time.perf_counter()
    fps_ema = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
        )

        bboxes = []
        probs_list = []

        if len(faces) > 0:
            batch_tensors = []
            for (x, y, w, h) in faces:
                bboxes.append((x, y, w, h))
                face_crop = gray_frame[y : y + h, x : x + w]
                batch_tensors.append(preprocess_live_frame(face_crop))

            batch_input = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                outputs = model(batch_input)
                batch_probs = F.softmax(outputs, dim=1).cpu().numpy()
                probs_list = [batch_probs[i] for i in range(len(bboxes))]

        track_ids = tracker.update(bboxes, probs_list)

        for (x, y, w, h), tid in zip(bboxes, track_ids):
            pred_idx, confidence = tracker.get_smoothed_prediction(tid)
            display_text = f"ID {tid} | {EMOTION_DICT[pred_idx]}: {confidence:.0f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                display_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        # Tính toán và hiển thị FPS theo thời gian thực (Exponential Moving Average)
        curr_time = time.perf_counter()
        instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        fps_ema = instant_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * instant_fps)

        cv2.putText(
            frame,
            f"FPS: {fps_ema:.1f} | Faces: {len(bboxes)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Real-time FER+ (Multi-Face Tracking)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_inference()