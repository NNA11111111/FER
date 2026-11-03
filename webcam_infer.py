from pathlib import Path
import time
import warnings
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import CustomCNN, FacePretrainedResNet18
from eval_utils import apply_posthoc_logit_adjustment
from preprocessing import PREPROCESSING_CONFIG, preprocess_fer_gray

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

DEFAULT_CLASS_PRIORS = np.array(
    [0.3600, 0.2815, 0.1093, 0.1257, 0.0888, 0.0059, 0.0226, 0.0063],
    dtype=np.float32,
)
DEFAULT_POSTHOC_TAU = 0.10


def shrink_face_bbox(x, y, w, h, frame_shape, crop_margin_shrink=0.0):
    """
    Tùy chọn thu nhỏ hộp giới hạn khuôn mặt Haar Cascade vào gần khung cắt kiểu FER2013.
    Mặc định crop_margin_shrink=0.0 giữ nguyên hộp Haar hiện tại.
    """
    shrink = float(crop_margin_shrink)
    if shrink <= 0.0:
        return x, y, w, h

    frame_h, frame_w = frame_shape[:2]
    dx = int(round(w * shrink * 0.5))
    dy = int(round(h * shrink * 0.5))

    nx = max(0, x + dx)
    ny = max(0, y + dy)
    nw = max(1, min(frame_w - nx, w - 2 * dx))
    nh = max(1, min(frame_h - ny, h - 2 * dy))
    return nx, ny, nw, nh


def preprocess_live_frame(face_crop):
    """
    Áp dụng chính xác hàm tiền xử lý dùng chung `preprocess_fer_gray_48x48` cho khung ảnh cắt từ Webcam:
    Grayscale -> Resize theo PREPROCESSING_CONFIG -> CLAHE + chuẩn hóa [-1, 1].
    """
    if len(face_crop.shape) == 3:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_crop

    X_norm = preprocess_fer_gray(face_gray, target_size=PREPROCESSING_CONFIG["target_size"])
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
    Tự động nạp trọng số cho CustomCNN bằng weights_only=True:
    Tương thích hoàn toàn với cả best_bundle.pth mới lẫn checkpoint best.pth cũ.
    """
    ckpt_data = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
        state_dict = ckpt_data["model_state_dict"]
        use_eca = bool(ckpt_data.get("use_eca", any(".eca." in k for k in state_dict.keys())))
        use_spatial = bool(ckpt_data.get("use_spatial_attn", any(".spatial_attn." in k for k in state_dict.keys())))
    else:
        state_dict = ckpt_data
        use_eca = any(".eca." in k for k in state_dict.keys())
        use_spatial = any(".spatial_attn." in k for k in state_dict.keys())

    if isinstance(ckpt_data, dict) and ckpt_data.get("model_type") == "face_resnet18":
        model = FacePretrainedResNet18(load_pretrained=False, num_classes=8)
    else:
        model = CustomCNN(num_classes=8, use_eca=use_eca, use_spatial_attn=use_spatial)
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt_data


def _validate_priors(priors):
    priors_arr = np.asarray(priors, dtype=np.float32).flatten()
    assert len(priors_arr) == 8, f"class_priors phải có đúng 8 phần tử, nhận được {len(priors_arr)}!"
    assert np.isclose(priors_arr.sum(), 1.0, atol=1e-2), (
        f"Tổng của class_priors phải xấp xỉ 1.0, nhận được {priors_arr.sum():.4f}!"
    )
    return priors_arr


def run_realtime_inference(
    bundle_path="best_bundle.pth",
    model_path="best.pth",
    calibration_path="best_calibration.pt",
    class_priors=None,
    posthoc_tau=None,
    temperature=1.0,
    crop_margin_shrink=0.0,
):
    """
    Chạy suy luận thời gian thực qua Webcam đồng bộ với quy trình đánh giá Single-Crop:
    - Ưu tiên nạp `best_bundle.pth` với `weights_only=True`.
    - Nếu không có bundle, nạp `best.pth` và `best_calibration.pt` (chỉ dùng `weights_only=True`);
      nếu lỗi hoặc thiếu calibration sẽ in CẢNH BÁO RÕ rồi dùng giá trị dự phòng.
    - Kiểm tra (assert) `class_priors` có đúng 8 phần tử và tổng ~ 1.0.
    - Hỗ trợ tham số tùy chọn `crop_margin_shrink` (mặc định 0.0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    print(f"Cấu hình crop_margin_shrink={crop_margin_shrink:.3f} (0.0 = giữ nguyên hộp Haar)")

    target_ckpt = bundle_path if Path(bundle_path).exists() else model_path

    try:
        model, ckpt_data = load_compatible_custom_cnn(target_ckpt, device)
        loaded_calib = False

        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            print(f"Đã nạp bundle checkpoint từ '{target_ckpt}' (weights_only=True).")
            if "class_priors" in ckpt_data and ("posthoc_tau_vec" in ckpt_data or "posthoc_tau" in ckpt_data):
                class_priors = ckpt_data["class_priors"]
                posthoc_tau = ckpt_data.get("posthoc_tau_vec", ckpt_data.get("posthoc_tau"))
                temperature = float(ckpt_data.get("temperature", temperature))
                loaded_calib = True

        if not loaded_calib and Path(calibration_path).exists():
            try:
                calib_data = torch.load(calibration_path, map_location="cpu", weights_only=True)
                if "class_priors" in calib_data and ("posthoc_tau" in calib_data or "posthoc_tau_vec" in calib_data):
                    class_priors = calib_data["class_priors"]
                    posthoc_tau = calib_data.get("posthoc_tau_vec", calib_data.get("posthoc_tau"))
                    temperature = float(calib_data.get("temperature", temperature))
                    loaded_calib = True
                    print(f"Đã nạp cấu hình hiệu chỉnh Logit từ '{calibration_path}' (weights_only=True).")
            except Exception as calib_err:
                warnings.warn(
                    f"\n[CẢNH BÁO RÕ] Không thể nạp '{calibration_path}' với weights_only=True ({calib_err}). "
                    "Bỏ qua file này và chuyển sang giá trị dự phòng an toàn."
                )
                print(
                    f"\n[CẢNH BÁO RÕ] Lỗi khi đọc '{calibration_path}' với weights_only=True: {calib_err}. "
                    "Đang chuyển sang dùng giá trị dự phòng."
                )

        if not loaded_calib:
            if class_priors is None or posthoc_tau is None:
                warnings.warn(
                    "\n[CẢNH BÁO RÕ] Không tìm thấy thông tin calibration hợp lệ (class_priors / posthoc_tau_vec) "
                    f"trong '{target_ckpt}' hoặc '{calibration_path}'! "
                    "Đang tạm dùng DEFAULT_CLASS_PRIORS và DEFAULT_POSTHOC_TAU dự phòng. "
                    "Hãy xuất best_bundle.pth từ notebook để đồng bộ chính xác với mô hình đã huấn luyện."
                )
                print(
                    "\n[CẢNH BÁO RÕ] Thiếu dữ liệu calibration hợp lệ! "
                    "Đang sử dụng DEFAULT_CLASS_PRIORS và DEFAULT_POSTHOC_TAU dự phòng."
                )
                if class_priors is None:
                    class_priors = DEFAULT_CLASS_PRIORS
                if posthoc_tau is None:
                    posthoc_tau = DEFAULT_POSTHOC_TAU

        class_priors = _validate_priors(class_priors)
        posthoc_tau = np.asarray(posthoc_tau, dtype=np.float32)

        model.to(device)
        model.eval()
        print("Tải trọng số và xác thực priors (8 lớp, tổng ~ 1.0) thành công. Đang mở Webcam...")
    except Exception as e:
        print(f"Lỗi: Không thể nạp mô hình từ '{target_ckpt}'.\nChi tiết: {e}")
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
                cx, cy, cw, ch = shrink_face_bbox(
                    x, y, w, h, gray_frame.shape, crop_margin_shrink=crop_margin_shrink
                )
                bboxes.append((cx, cy, cw, ch))
                face_crop = gray_frame[cy : cy + ch, cx : cx + cw]
                batch_tensors.append(preprocess_live_frame(face_crop))

            batch_input = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                outputs = model(batch_input)
                adj_outputs = apply_posthoc_logit_adjustment(
                    outputs, class_priors=class_priors, tau=posthoc_tau
                )
                batch_probs = F.softmax(adj_outputs / max(float(temperature), 1e-6), dim=1).cpu().numpy()
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
