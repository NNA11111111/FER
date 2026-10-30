import hashlib
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EMOTION_COLS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

LABEL_NAMES = [
    "Neutral", "Happiness", "Surprise", "Sadness",
    "Anger", "Disgust", "Fear", "Contempt",
]


def load_fer2013_csv(
    img_csv_path,
    label_csv_path,
    return_soft_labels=False,
    use_stratify=True,
    drop_ties=True,
    random_state=42,
):
    """
    Đọc và tiền xử lý bộ dữ liệu FER2013 kết hợp nhãn đám đông FER+ (Barsoum et al., 2016).
    Cải tiến toàn diện:
    1. Khử trùng lặp chéo (Deduplication) theo MD5 hash của chuỗi pixel gốc trước khi chia tập,
       ngăn rò rỉ dữ liệu giữa Train / Validation / Test.
    2. Lọc bỏ các ảnh có phiếu bầu cao nhất bị hoà (ties, khi drop_ties=True) để nhãn không bị thiên lệch về Neutral.
    3. Chia tập phân tầng (Stratified Split 80/10/10) bằng `train_test_split(..., stratify=y_hard)`
       nhằm giữ nguyên tỷ lệ 8 lớp cảm xúc (đặc biệt là Disgust và Contempt) trên cả 3 tập.
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
        if len(df_img) != len(df_label):
            raise ValueError(
                f"Không thể ghép dữ liệu: số dòng không khớp (ảnh={len(df_img)}, nhãn={len(df_label)})."
            )
        df = df_label.copy()
        df["pixels"] = df_img["pixels"].to_numpy()
        print("Ghép ảnh và nhãn theo thứ tự dòng (hai file không có ID chung).")

    required = {"Usage", "pixels", *EMOTION_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    # --- BƯỚC 1: KHỬ TRÙNG LẶP THEO HASH ẢNH ---
    df["img_hash"] = df["pixels"].apply(
        lambda p: hashlib.md5(str(p).strip().encode("utf-8")).hexdigest()
    )
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["img_hash"], keep="first").reset_index(drop=True)
    dropped_duplicates = before_dedup - len(df)

    ignored_usage = 0
    dropped_no_votes = 0
    dropped_unknown_nf = 0
    dropped_ties = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    valid_usages = {"Training", "PublicTest", "PrivateTest"}

    X_all, y_soft_all, y_hard_all, usage_all = [], [], [], []

    for _, row in df.iterrows():
        usage = row["Usage"]
        if usage not in valid_usages:
            ignored_usage += 1
            continue

        votes = row[EMOTION_COLS].to_numpy(dtype=np.float32).copy()
        unknown_votes = int(row["unknown"]) if "unknown" in df.columns else 0
        nf_votes = int(row["NF"]) if "NF" in df.columns else 0

        # Lọc nhiễu theo Barsoum et al., 2016: bỏ các lớp chỉ có <= 1 phiếu
        votes[votes <= 1] = 0.0

        total_votes = votes.sum()
        if total_votes == 0:
            dropped_no_votes += 1
            continue

        max_vote = votes.max()

        # Bỏ ảnh nếu unknown hoặc NF >= cảm xúc cao nhất
        if max_vote <= unknown_votes or max_vote <= nf_votes:
            dropped_unknown_nf += 1
            continue

        # --- BƯỚC 2: XỬ LÝ ẢNH HOÀ PHIẾU CAO NHẤT (TIES) ---
        if np.sum(votes == max_vote) > 1:
            if drop_ties:
                dropped_ties += 1
                continue

        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(f"Một ảnh có {pixels.size} pixel; cần đúng {48 * 48}.")

        raw_img = pixels.reshape(48, 48)
        image = clahe.apply(raw_img)
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        image = image[..., np.newaxis]

        soft_probs = votes / total_votes
        hard_label = int(np.argmax(votes))

        X_all.append(image)
        y_soft_all.append(soft_probs)
        y_hard_all.append(hard_label)
        usage_all.append(usage)

    X_all = np.asarray(X_all, dtype=np.float32)
    y_soft_all = np.asarray(y_soft_all, dtype=np.float32)
    y_hard_all = np.asarray(y_hard_all, dtype=np.int64)
    usage_all = np.asarray(usage_all)

    # --- BƯỚC 3: CHIA TẬP VỚI STRATIFY HOẶC THEO USAGE GỐC ---
    if use_stratify:
        # Bước 3a: Chia 80% Train và 20% Temp (Val + Test) có phân tầng theo y_hard_all
        (
            X_train, X_temp,
            y_train_soft, y_temp_soft,
            y_train_hard, y_temp_hard,
        ) = train_test_split(
            X_all, y_soft_all, y_hard_all,
            test_size=0.20,
            random_state=random_state,
            stratify=y_hard_all,
        )
        # Bước 3b: Chia đôi Temp (50/50) thành 10% Validation và 10% Test có phân tầng theo y_temp_hard
        (
            X_val, X_test,
            y_val_soft, y_test_soft,
            y_val_hard, y_test_hard,
        ) = train_test_split(
            X_temp, y_temp_soft, y_temp_hard,
            test_size=0.50,
            random_state=random_state,
            stratify=y_temp_hard,
        )
        split_method_str = f"Stratified Split (80/10/10, random_state={random_state})"
    else:
        train_mask = usage_all == "Training"
        val_mask = usage_all == "PublicTest"
        test_mask = usage_all == "PrivateTest"

        X_train, y_train_soft, y_train_hard = X_all[train_mask], y_soft_all[train_mask], y_hard_all[train_mask]
        X_val, y_val_soft, y_val_hard = X_all[val_mask], y_soft_all[val_mask], y_hard_all[val_mask]
        X_test, y_test_soft, y_test_hard = X_all[test_mask], y_soft_all[test_mask], y_hard_all[test_mask]
        split_method_str = "Chia theo cột Usage gốc (Training / PublicTest / PrivateTest)"

    def print_split_stats(split_name, labels):
        counts = np.bincount(labels, minlength=len(EMOTION_COLS))
        distribution = ", ".join(
            f"{LABEL_NAMES[i]}={counts[i]:,} ({counts[i]/len(labels)*100:.1f}%)"
            for i in range(len(LABEL_NAMES))
        )
        print(f"{split_name}: {len(labels):,} ảnh")
        print(f"  Phân phối: {distribution}")

    print("\n=== KẾT QUẢ KHỬ TRÙNG, LỌC NHIỄU VÀ CHIA TẬP ===")
    print(f"Phương pháp chia tập: {split_method_str}")
    print(f"Bị loại do trùng hash pixel (Deduplication): {dropped_duplicates:,}")
    print(f"Bỏ qua do Usage không thuộc 3 tập: {ignored_usage:,}")
    print(f"Bị loại vì không còn phiếu cảm xúc > 1: {dropped_no_votes:,}")
    print(f"Bị loại vì unknown/NF >= cảm xúc cao nhất: {dropped_unknown_nf:,}")
    print(f"Bị loại vì hoà phiếu cao nhất (Ties): {dropped_ties:,}")
    print(f"Tổng ảnh sạch được giữ trong 3 tập: {len(y_hard_all):,}\n")

    print_split_stats("Train", y_train_hard)
    print_split_stats("Validation", y_val_hard)
    print_split_stats("Test", y_test_hard)

    if return_soft_labels:
        return (
            (X_train, y_train_soft, y_train_hard),
            (X_val, y_val_soft, y_val_hard),
            (X_test, y_test_soft, y_test_hard),
        )

    return (
        (X_train, y_train_hard),
        (X_val, y_val_hard),
        (X_test, y_test_hard),
    )