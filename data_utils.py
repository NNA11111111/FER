from pathlib import Path

import cv2
import numpy as np
import pandas as pd


EMOTION_COLS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]


def load_fer2013_csv(img_csv_path, label_csv_path):
    df_img = pd.read_csv(img_csv_path)
    df_label = pd.read_csv(label_csv_path)

    # Ghép bằng ID nếu cả hai file đều có cột ID chung.
    common_id_cols = [
        col for col in ("image_id", "id")
        if col in df_img.columns and col in df_label.columns
    ]

    if common_id_cols:
        id_col = common_id_cols[0]
        if df_img[id_col].duplicated().any() or df_label[id_col].duplicated().any():
            raise ValueError(f"Cột ID '{id_col}' có giá trị trùng.")

        df = df_label.merge(
            df_img[[id_col, "pixels"]],
            on=id_col,
            how="left",
            validate="one_to_one",
        )
        if df["pixels"].isna().any():
            raise ValueError("Một số nhãn không tìm thấy ảnh tương ứng.")
    else:
        # Không có ID: chỉ có thể kiểm tra số dòng, không xác minh được thứ tự.
        if len(df_img) != len(df_label):
            raise ValueError(
                f"Số dòng không khớp: ảnh={len(df_img)}, nhãn={len(df_label)}."
            )
        df = df_label.copy()
        df["pixels"] = df_img["pixels"].to_numpy()

    required = {"Usage", "pixels", *EMOTION_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    for _, row in df.iterrows():
        votes = row[EMOTION_COLS].to_numpy(dtype=int).copy()
        unknown_votes = int(row["unknown"]) if "unknown" in df.columns else 0
        nf_votes = int(row["NF"]) if "NF" in df.columns else 0

        votes[votes <= 1] = 0
        if votes.sum() == 0:
            continue

        max_vote = votes.max()
        if max_vote <= unknown_votes or max_vote <= nf_votes:
            continue

        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(f"Ảnh có {pixels.size} pixel; cần đúng {48 * 48}.")

        image = pixels.reshape(48, 48)
        image = clahe.apply(image)
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        image = image[..., np.newaxis]
        label = int(np.argmax(votes))

        usage = row["Usage"]
        if usage == "Training":
            X_train.append(image)
            y_train.append(label)
        elif usage == "PublicTest":
            X_val.append(image)
            y_val.append(label)
        elif usage == "PrivateTest":
            X_test.append(image)
            y_test.append(label)
        # Các giá trị khác, gồm "Other", không đưa vào ba tập này.

    return (
        (np.asarray(X_train), np.asarray(y_train)),
        (np.asarray(X_val), np.asarray(y_val)),
        (np.asarray(X_test), np.asarray(y_test)),
    )