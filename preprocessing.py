import cv2
import numpy as np

PREPROCESSING_CONFIG = {
    "target_size": [48, 48],
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": [4, 4],
    "norm_mean": 0.5,
    "norm_std": 0.5,
}


def preprocess_fer_gray_48x48(
    gray_img_48x48: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (4, 4),
) -> np.ndarray:
    """
    Hàm tiền xử lý dùng chung cho cả pipeline huấn luyện (data_utils.py)
    và suy luận thời gian thực (webcam_infer.py).

    Đầu vào:
        gray_img_48x48: Ảnh xám 2D kích thước (48, 48), kiểu uint8.
    Đầu ra:
        Ảnh float32 kích thước (48, 48) đã qua CLAHE và chuẩn hóa về miền [-1.0, 1.0].
    """
    if not isinstance(gray_img_48x48, np.ndarray):
        gray_img_48x48 = np.asarray(gray_img_48x48, dtype=np.uint8)

    if gray_img_48x48.ndim != 2 or gray_img_48x48.shape != (48, 48):
        raise ValueError(
            f"Ảnh đầu vào phải là ảnh xám 2D kích thước (48, 48), nhận được shape={gray_img_48x48.shape}."
        )
    if gray_img_48x48.dtype != np.uint8:
        raise ValueError(
            f"Ảnh đầu vào phải có kiểu uint8, nhận được dtype={gray_img_48x48.dtype}."
        )

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    img_clahe = clahe.apply(gray_img_48x48)
    img_norm = (img_clahe.astype(np.float32) / 255.0 - 0.5) / 0.5
    return img_norm