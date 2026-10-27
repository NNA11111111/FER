import cv2
import numpy as np
import pandas as pd


EMOTION_COLS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]


def load_fer2013_csv(img_csv_path, label_csv_path, return_soft_labels=False):
    """
    Đọc và tiền xử lý bộ dữ liệu FER2013 kết hợp nhãn đám đông FER+ (Barsoum et al., 2016).
    - Nếu return_soft_labels=False: trả về (X, y_hard) như cũ để tương thích ngược.
    - Nếu return_soft_labels=True: trả về (X, y_soft, y_hard) trong đó:
        + y_soft: phân phối xác suất 8 cảm xúc (float32, tổng = 1.0) dùng để train với Soft Label / KLDivLoss.
        + y_hard: nhãn đa số (int64) dùng để tính Accuracy, Macro F1, Weighted F1 và Sampler.
    """
    df_img = pd.read_csv(img_csv_path)
    df_label = pd.read_csv(label_csv_path)

    print("=== THỐNG KÊ DỮ LIỆU ĐẦU VÀO ===")
    print(f"Ảnh trong fer2013.csv: {len(df_img):,} dòng")
    print(f"Nhãn trong fer2013new.csv: {len(df_label):,} dòng")

    # Ghép bằng ID nếu cả hai file có cột ID chung.
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

        print(f"Đã ghép ảnh và nhãn bằng cột ID: {id_col}")
    else:
        # Không có ID chung: ghép theo thứ tự dòng.
        if len(df_img) != len(df_label):
            raise ValueError(
                "Không thể ghép dữ liệu: số dòng không khớp "
                f"(ảnh={len(df_img)}, nhãn={len(df_label)})."
            )

        df = df_label.copy()
        df["pixels"] = df_img["pixels"].to_numpy()

        print("Ghép ảnh và nhãn theo thứ tự dòng (hai file không có ID chung).")

    required = {"Usage", "pixels", *EMOTION_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    X_train, y_train_soft, y_train_hard = [], [], []
    X_val, y_val_soft, y_val_hard = [], [], []
    X_test, y_test_soft, y_test_hard = [], [], []

    ignored_usage = 0
    dropped_no_votes = 0
    dropped_unknown_nf = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    valid_usages = {"Training", "PublicTest", "PrivateTest"}

    for _, row in df.iterrows():
        usage = row["Usage"]

        # Không đưa Other hoặc Usage lạ vào train/validation/test.
        if usage not in valid_usages:
            ignored_usage += 1
            continue

        # FER+ cung cấp phiếu bầu cho 8 cảm xúc.
        votes = row[EMOTION_COLS].to_numpy(dtype=np.float32).copy()
        unknown_votes = int(row["unknown"]) if "unknown" in df.columns else 0
        nf_votes = int(row["NF"]) if "NF" in df.columns else 0

        # Bỏ các cảm xúc chỉ có 0 hoặc 1 phiếu (lọc nhiễu theo Barsoum et al., 2016).
        votes[votes <= 1] = 0.0

        total_votes = votes.sum()
        if total_votes == 0:
            dropped_no_votes += 1
            continue

        max_vote = votes.max()

        # Bỏ ảnh nếu unknown hoặc NF có số phiếu bằng/vượt cảm xúc cao nhất.
        if max_vote <= unknown_votes or max_vote <= nf_votes:
            dropped_unknown_nf += 1
            continue

        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(
                f"Một ảnh có {pixels.size} pixel; cần đúng {48 * 48}."
            )

        image = pixels.reshape(48, 48)
        image = clahe.apply(image)
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        image = image[..., np.newaxis]

        # 1. Phân phối xác suất nhãn mềm (Soft Label Distribution - Barsoum et al., 2016)
        soft_probs = votes / total_votes
        # 2. Nhãn cứng (Majority Voting) để phục vụ thống kê và đánh giá F1-Score
        hard_label = int(np.argmax(votes))

        if usage == "Training":
            X_train.append(image)
            y_train_soft.append(soft_probs)
            y_train_hard.append(hard_label)
        elif usage == "PublicTest":
            X_val.append(image)
            y_val_soft.append(soft_probs)
            y_val_hard.append(hard_label)
        else:  # PrivateTest
            X_test.append(image)
            y_test_soft.append(soft_probs)
            y_test_hard.append(hard_label)

    label_names = [
        "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt",
    ]

    def print_split_stats(split_name, labels):
        counts = np.bincount(labels, minlength=len(EMOTION_COLS))
        distribution = ", ".join(
            f"{label_names[i]}={counts[i]:,}"
            for i in range(len(label_names))
        )
        print(f"{split_name}: {len(labels):,} ảnh")
        print(f"  Nhãn đa số: {distribution}")

    kept_count = len(y_train_hard) + len(y_val_hard) + len(y_test_hard)

    print("\n=== KẾT QUẢ LỌC VÀ CHIA TẬP ===")
    print(f"Bỏ qua do Usage không thuộc 3 tập: {ignored_usage:,}")
    print(f"Bị loại vì không còn phiếu cảm xúc > 1: {dropped_no_votes:,}")
    print(f"Bị loại vì unknown/NF >= cảm xúc cao nhất: {dropped_unknown_nf:,}")
    print(f"Tổng ảnh được giữ trong 3 tập: {kept_count:,}")

    print_split_stats("Train", y_train_hard)
    print_split_stats("Validation", y_val_hard)
    print_split_stats("Test", y_test_hard)

    if return_soft_labels:
        return (
            (np.asarray(X_train), np.asarray(y_train_soft, dtype=np.float32), np.asarray(y_train_hard, dtype=np.int64)),
            (np.asarray(X_val), np.asarray(y_val_soft, dtype=np.float32), np.asarray(y_val_hard, dtype=np.int64)),
            (np.asarray(X_test), np.asarray(y_test_soft, dtype=np.float32), np.asarray(y_test_hard, dtype=np.int64)),
        )

    return (
        (np.asarray(X_train), np.asarray(y_train_hard, dtype=np.int64)),
        (np.asarray(X_val), np.asarray(y_val_hard, dtype=np.int64)),
        (np.asarray(X_test), np.asarray(y_test_hard, dtype=np.int64)),
    )