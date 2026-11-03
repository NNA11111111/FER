import cv2
import numpy as np

PREPROCESSING_CONFIG = {
    "target_size": [112, 112],
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": [4, 4],
    "norm_mean": 0.5,
    "norm_std": 0.5,
}


def preprocess_fer_gray(
    gray_img: np.ndarray,
    target_size: tuple = (112, 112),
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (4, 4),
) -> np.ndarray:
    """
    Chuyển ảnh xám uint8 về kích thước đích, áp dụng CLAHE và chuẩn hóa [-1, 1].
    Dùng chung cho pipeline dữ liệu, suy luận webcam và tái dựng split manifest.

    Đầu vào:
        gray_img: Ảnh xám 2D kiểu uint8.
        target_size: Kích thước (width, height) theo quy ước OpenCV.
    Đầu ra:
        Ảnh float32 kích thước (height, width), đã qua CLAHE và chuẩn hóa [-1, 1].
    """
    if not isinstance(gray_img, np.ndarray):
        gray_img = np.asarray(gray_img, dtype=np.uint8)

    if gray_img.ndim != 2 or gray_img.size == 0:
        raise ValueError(
            f"Ảnh đầu vào phải là ảnh xám 2D không rỗng, nhận được shape={gray_img.shape}."
        )
    if gray_img.dtype != np.uint8:
        raise ValueError(
            f"Ảnh đầu vào phải có kiểu uint8, nhận được dtype={gray_img.dtype}."
        )
    if len(target_size) != 2 or any(int(dimension) <= 0 for dimension in target_size):
        raise ValueError(f"target_size phải gồm (width, height) dương, nhận được {target_size}.")

    target_size = tuple(int(dimension) for dimension in target_size)
    if (gray_img.shape[1], gray_img.shape[0]) != target_size:
        shrinking = gray_img.shape[1] > target_size[0] or gray_img.shape[0] > target_size[1]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        gray_img = cv2.resize(gray_img, target_size, interpolation=interpolation)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    img_clahe = clahe.apply(gray_img)
    img_norm = (img_clahe.astype(np.float32) / 255.0 - 0.5) / 0.5
    return img_norm


def preprocess_fer_gray_48x48(
    gray_img_48x48: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (4, 4),
) -> np.ndarray:
    """Backward-compatible preprocessing for an already resized 48×48 grayscale image."""
    if not isinstance(gray_img_48x48, np.ndarray):
        gray_img_48x48 = np.asarray(gray_img_48x48, dtype=np.uint8)
    if gray_img_48x48.ndim != 2 or gray_img_48x48.shape != (48, 48):
        raise ValueError(
            f"Ảnh đầu vào phải là ảnh xám 2D kích thước (48, 48), nhận được shape={gray_img_48x48.shape}."
        )
    return preprocess_fer_gray(
        gray_img_48x48,
        target_size=(48, 48),
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
