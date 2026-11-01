from pathlib import Path
import time
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import CustomCNN
from eval_utils import apply_posthoc_logit_adjustment

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

# Phân phối tiên nghiệm chuẩn (Clean Stratified Class Priors pi_c) từ tập FER+ 70/15/15
DEFAULT_CLASS_PRIORS = np.array(
    [0.3600, 0.2815, 0.1093, 0.1257, 0.0888, 0.0059, 0.0226, 0.0063],
    dtype=np.float32,
)
# Mặc định chỉ bù nhẹ 0.10 ở bước hậu nghiệm vì mô hình đã tích hợp stage1_tau=0.30 khi huấn luyện
DEFAULT_POSTHOC_TAU = 0.10


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
    giữa các frame liên tiếp, giúp dự đoán ổn định khi có nhiều người trong khung hình.
    """
    def __init__(self, maxlen=5, max_distance=80.0, max_missed_frames=10):
        self.maxlen = maxlen
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.tracks = {}
        self.next_id = 1

    def update(self, bboxes, probs_list):
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

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed_frames:
                del self.tracks[tid]

        return assigned_track_ids

    def get_smoothed_prediction(self, track_id):
        avg_probs = np.mean(self.tracks[track_id]["queue"], axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx]) * 100.0
        return pred_idx, confidence


def load_compatible_custom_cnn(model_path, device):
    """
    Tự động nạp trọng số cho CustomCNN: tương thích hoàn toàn với cả checkpoint mới
    (có ECA + SpatialMicroAttention) lẫn checkpoint best.pth phiên bản cũ (chỉ có ResSEBlock).
    """
    ckpt_data = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt_data["model_state_dict"] if (isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data) else ckpt_data

    has_eca = any(".eca." in k for k in state_dict.keys())
    has_spatial = any(".spatial_attn." in k for k in state_dict.keys())

    model = CustomCNN(num_classes=8, use_eca=has_eca, use_spatial_attn=has_spatial)
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt_data


def run_realtime_inference(
    model_path="best.pth",
    calibration_path="best_calibration.pt",
    class_priors=DEFAULT_CLASS_PRIORS,
    posthoc_tau=DEFAULT_POSTHOC_TAU,
):
    """
    Chạy suy luận thời gian thực qua Webcam đồng bộ 100% với quy trình đánh giá (Evaluation).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    try:
        model, ckpt_data = load_compatible_custom_cnn(model_path, device)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            class_priors = ckpt_data.get("class_priors", class_priors)
            posthoc_tau = ckpt_data.get("posthoc_tau", posthoc_tau)

        if Path(calibration_path).exists():
            calib_data = torch.load(calibration_path, map_location="cpu", weights_only=False)
            class_priors = calib_data.get("class_priors", class_priors)
            posthoc_tau = calib_data.get("posthoc_tau", posthoc_tau)
            print(f"Đã nạp cấu hình hiệu chỉnh Logit từ '{calibration_path}'.")

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
                adj_outputs = apply_posthoc_logit_adjustment(
                    outputs, class_priors=class_priors, tau=posthoc_tau
                )
                batch_probs = F.softmax(adj_outputs, dim=1).cpu().numpy()
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

        cv2.imshow("Real-time FER+ (CustomCNN + ECA/Spatial + Logit Adjustment)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_inference()