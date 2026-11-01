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


def compute_dhash(img_2d, hash_size=8):
    """
    Tính Perceptual Difference Hash (dHash 64-bit) cho ảnh xám 2D.
    Giúp phát hiện các ảnh gần trùng (near-duplicates) do nén JPEG, chỉnh sáng nhẹ
    hoặc dịch chuyển 1-2 pixel mà hàm băm mật mã MD5 bỏ sót.
    """
    resized = cv2.resize(img_2d, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    hash_val = 0
    for b in bits:
        hash_val = (hash_val << 1) | int(b)
    return f"{hash_val:016x}"


def compute_crowd_sample_weight(soft_probs, valid_votes_sum, unknown_votes=0, nf_votes=0, is_rescued_tie=False, min_weight=0.25):
    """
    Tính Trọng số Độ tin cậy của Mẫu (Sample Weight w_i in [min_weight, 1.0]) trực tiếp từ
    phân phối 10 phiếu bầu của FER+ (Barsoum et al., ICMI 2016) và nguyên lý Sample Weighting:
    - Thành phần 1 (Độ rõ ràng biểu cảm - Entropy): E_norm = -sum(p_c * log(p_c)) / log(8).
      Các ảnh đồng thuận cao (entropy thấp) nhận trọng số xấp xỉ 1.0; ảnh phân tán nhiều lớp nhận trọng số thấp hơn.
    - Thành phần 2 (Tỷ lệ phiếu hợp lệ so với phiếu nhiễu unknown/NF):
      valid_ratio = valid_votes_sum / (valid_votes_sum + unknown_votes + nf_votes).
    - Nếu là mẫu hoà phiếu được thu hồi riêng cho lớp hiếm (is_rescued_tie=True), nhân hệ số giảm 0.35
      để tránh làm nhiễu biên quyết định của lớp hiếm với lớp đa số.
    """
    probs = np.clip(np.asarray(soft_probs, dtype=np.float64), 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(EMOTION_COLS))
    clarity_score = 1.0 - 0.45 * (entropy / max_entropy)

    total_raw = float(valid_votes_sum + unknown_votes + nf_votes)
    valid_ratio = float(valid_votes_sum) / max(total_raw, 1.0)

    weight = clarity_score * valid_ratio
    if is_rescued_tie:
        weight *= 0.35

    return float(np.clip(weight, min_weight, 1.0))


def build_multilabel_stratify_keys(y_soft, y_hard, min_count_per_key=4):
    """
    Phân tầng đa nhãn (Multi-label / Iterative Stratification theo tinh thần Sechidis et al., 2011):
    Kết hợp cặp (nhãn_chính, nhãn_phụ_lớn_nhất_có_phiếu >= 2, tức xác suất >= 0.15)
    để bảo toàn đồng thời cả cảm xúc chính lẫn cảm xúc phụ giữa 3 tập Train / Val / Test.
    Nếu một tổ hợp cặp quá hiếm (< min_count_per_key mẫu), tự động quy về nhãn chính (y_hard)
    để tránh lỗi ValueError khi chia train_test_split.
    """
    raw_keys = []
    for probs, primary in zip(y_soft, y_hard):
        sec_probs = probs.copy()
        sec_probs[primary] = 0.0
        sec_idx = int(np.argmax(sec_probs))
        # Ngưỡng 0.15 tương ứng với >= 2 phiếu bầu hợp lệ trên thang ~10 phiếu của FER+
        if sec_probs[sec_idx] >= 0.15:
            raw_keys.append(f"{primary}_{sec_idx}")
        else:
            raw_keys.append(f"{primary}_none")

    key_series = pd.Series(raw_keys)
    key_counts = key_series.value_counts()

    final_keys = [
        k if key_counts[k] >= min_count_per_key else f"{primary}_none"
        for k, primary in zip(raw_keys, y_hard)
    ]

    # Kiểm tra an toàn lần 2 sau khi gộp
    final_series = pd.Series(final_keys)
    final_counts = final_series.value_counts()
    safe_keys = [
        k if final_counts[k] >= min_count_per_key else str(primary)
        for k, primary in zip(final_keys, y_hard)
    ]
    return np.asarray(safe_keys)


def load_fer2013_csv(
    img_csv_path,
    label_csv_path,
    return_soft_labels=False,
    return_sample_weights=False,
    use_stratify=True,
    use_multilabel_stratify=True,
    use_perceptual_dedup=True,
    drop_ties=True,
    retain_rare_ties_in_train=False,
    rare_class_indices=(5, 6, 7),
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Đọc và tiền xử lý bộ dữ liệu FER2013 kết hợp nhãn đám đông FER+ (Barsoum et al., ICMI 2016).
    Cải tiến chống nhiễu biên lớp hiếm (Disgust, Fear, Contempt):
    1. Khử trùng lặp 2 tầng (MD5 Exact Hash + Perceptual dHash 64-bit) trước khi chia tập,
       ngăn chặn 100% rò rỉ dữ liệu giữa Train / Val / Test.
    2. Lọc bỏ các ảnh rác/mất khuôn mặt dựa trên phiếu NF/unknown của 10 chuyên gia gán nhãn FER+,
       kết hợp tính Trọng số Độ tin cậy của Mẫu (sample_weights w_i in [0.25, 1.0]).
    3. Mặc định TẮT `retain_rare_ties_in_train=False` để ngăn chặn việc bơm 161 ảnh hoà phiếu mơ hồ
       (50% Neutral - 50% Contempt hoặc 50% Anger - 50% Disgust) vào tập Train. Nhờ đó, tập Train
       chỉ học các mẫu đơn trị có độ đồng thuận rõ ràng, giúp tăng mạnh Precision & F1 của Disgust,
       Contempt và Fear trên tập Test.
    4. Chia tập bằng Stratified Split 2 bước theo tỷ lệ 70% Train / 15% Validation / 15% Test
       với khóa phân tầng đa nhãn (Nhãn chính + Nhãn phụ).
    """
    df_img = pd.read_csv(img_csv_path)
    df_label = pd.read_csv(label_csv_path)

    print("=== THỐNG KÊ DỮ LIỆU ĐẦU VÀO ===")
    print(f"Ảnh trong fer2013.csv: {len(df_img):,} dòng")
    print(f"Nhãn trong fer2013new.csv: {len(df_label):,} dòng")

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

    # --- BƯỚC 1A: KHỬ TRÙNG LẶP TUYỆT ĐỐI THEO MD5 HASH ---
    df["img_hash"] = df["pixels"].apply(
        lambda p: hashlib.md5(str(p).strip().encode("utf-8")).hexdigest()
    )
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["img_hash"], keep="first").reset_index(drop=True)
    dropped_md5_duplicates = before_dedup - len(df)

    ignored_usage = 0
    dropped_no_votes = 0
    dropped_unknown_nf = 0
    dropped_ties = 0
    rescued_rare_ties = 0
    dropped_phash_duplicates = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    valid_usages = {"Training", "PublicTest", "PrivateTest"}
    seen_phashes = set()
    rare_set = set(rare_class_indices) if rare_class_indices is not None else set()

    X_all, y_soft_all, y_hard_all, w_all, usage_all = [], [], [], [], []
    X_rare_ties, y_soft_rare_ties, y_hard_rare_ties, w_rare_ties = [], [], [], []

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

        # Bỏ ảnh rác / che khuất nặng nếu unknown hoặc NF >= cảm xúc cao nhất
        if max_vote <= unknown_votes or max_vote <= nf_votes:
            dropped_unknown_nf += 1
            continue

        # --- BƯỚC 2: XỬ LÝ ẢNH HOÀ PHIẾU CAO NHẤT (TIES) ---
        tied_classes = np.where(votes == max_vote)[0]
        is_tie = len(tied_classes) > 1
        tied_rare_classes = [int(c) for c in tied_classes if int(c) in rare_set]

        rescue_tie_for_train = (
            usage == "Training"
            and retain_rare_ties_in_train
            and len(tied_rare_classes) > 0
            and max_vote >= 3
        )
        if is_tie and drop_ties:
            if not rescue_tie_for_train:
                dropped_ties += 1
                continue

        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(f"Một ảnh có {pixels.size} pixel; cần đúng {48 * 48}.")

        raw_img = pixels.reshape(48, 48)

        # --- BƯỚC 1B: KHỬ TRÙNG LẶP THỊ GIÁC (PERCEPTUAL DHASH) ---
        if use_perceptual_dedup:
            phash = compute_dhash(raw_img)
            if phash in seen_phashes:
                dropped_phash_duplicates += 1
                continue
            seen_phashes.add(phash)

        image = clahe.apply(raw_img)
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        image = image[..., np.newaxis]

        soft_probs = votes / total_votes

        if is_tie and drop_ties and rescue_tie_for_train:
            hard_label = int(tied_rare_classes[-1])
            sample_w = compute_crowd_sample_weight(
                soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=True, min_weight=0.20
            )
            X_rare_ties.append(image)
            y_soft_rare_ties.append(soft_probs)
            y_hard_rare_ties.append(hard_label)
            w_rare_ties.append(sample_w)
            rescued_rare_ties += 1
        else:
            hard_label = int(np.argmax(votes))
            sample_w = compute_crowd_sample_weight(
                soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=False
            )
            X_all.append(image)
            y_soft_all.append(soft_probs)
            y_hard_all.append(hard_label)
            w_all.append(sample_w)
            usage_all.append(usage)

    X_all = np.asarray(X_all, dtype=np.float32)
    y_soft_all = np.asarray(y_soft_all, dtype=np.float32)
    y_hard_all = np.asarray(y_hard_all, dtype=np.int64)
    w_all = np.asarray(w_all, dtype=np.float32)
    usage_all = np.asarray(usage_all)

    # --- BƯỚC 3: CHIA TẬP VỚI PHÂN TẦNG ĐA NHÃN STRATIFIED (70/15/15 HOẶC TÙY CHỈNH) ---
    if use_stratify:
        holdout_size = val_size + test_size
        if not (0.0 < holdout_size < 1.0):
            raise ValueError(f"Tổng val_size + test_size phải nằm trong (0, 1), nhận được {holdout_size}.")

        if use_multilabel_stratify:
            stratify_keys_all = build_multilabel_stratify_keys(
                y_soft_all, y_hard_all, min_count_per_key=max(4, int(round(2.0 / min(val_size, test_size))))
            )
            strat_desc = "Multi-label Composite Stratification (Nhãn chính + Nhãn phụ)"
        else:
            stratify_keys_all = y_hard_all
            strat_desc = "Single-label Stratification (Nhãn cứng y_hard)"

        # Bước 3a: Tách Train (70%) và Temp (30% = Val + Test) trên tập mẫu đơn trị sạch
        (
            X_train, X_temp,
            y_train_soft, y_temp_soft,
            y_train_hard, y_temp_hard,
            w_train, w_temp,
            _, temp_keys,
        ) = train_test_split(
            X_all, y_soft_all, y_hard_all, w_all, stratify_keys_all,
            test_size=holdout_size,
            random_state=random_state,
            stratify=stratify_keys_all,
        )

        # Kiểm tra an toàn cho khóa phân tầng trên tập Temp trước khi chia đôi Val / Test
        temp_key_counts = pd.Series(temp_keys).value_counts()
        safe_temp_keys = np.asarray([
            k if temp_key_counts[k] >= 2 else str(h)
            for k, h in zip(temp_keys, y_temp_hard)
        ])

        test_ratio_in_temp = test_size / holdout_size
        (
            X_val, X_test,
            y_val_soft, y_test_soft,
            y_val_hard, y_test_hard,
            w_val, w_test,
        ) = train_test_split(
            X_temp, y_temp_soft, y_temp_hard, w_temp,
            test_size=test_ratio_in_temp,
            random_state=random_state,
            stratify=safe_temp_keys,
        )
        train_pct = int(round((1.0 - holdout_size) * 100))
        val_pct = int(round(val_size * 100))
        test_pct = int(round(test_size * 100))
        split_method_str = f"{strat_desc} ({train_pct}/{val_pct}/{test_pct}, random_state={random_state})"
    else:
        train_mask = usage_all == "Training"
        val_mask = usage_all == "PublicTest"
        test_mask = usage_all == "PrivateTest"

        X_train, y_train_soft, y_train_hard, w_train = (
            X_all[train_mask], y_soft_all[train_mask], y_hard_all[train_mask], w_all[train_mask]
        )
        X_val, y_val_soft, y_val_hard, w_val = (
            X_all[val_mask], y_soft_all[val_mask], y_hard_all[val_mask], w_all[val_mask]
        )
        X_test, y_test_soft, y_test_hard, w_test = (
            X_all[test_mask], y_soft_all[test_mask], y_hard_all[test_mask], w_all[test_mask]
        )
        split_method_str = "Chia theo cột Usage gốc (Training / PublicTest / PrivateTest)"

    y_train_stratified_clean = y_train_hard.copy()

    if len(X_rare_ties) > 0:
        X_rare_arr = np.asarray(X_rare_ties, dtype=np.float32)
        y_soft_rare_arr = np.asarray(y_soft_rare_ties, dtype=np.float32)
        y_hard_rare_arr = np.asarray(y_hard_rare_ties, dtype=np.int64)
        w_rare_arr = np.asarray(w_rare_ties, dtype=np.float32)

        X_train = np.concatenate([X_train, X_rare_arr], axis=0)
        y_train_soft = np.concatenate([y_train_soft, y_soft_rare_arr], axis=0)
        y_train_hard = np.concatenate([y_train_hard, y_hard_rare_arr], axis=0)
        w_train = np.concatenate([w_train, w_rare_arr], axis=0)

        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train_soft = y_train_soft[perm]
        y_train_hard = y_train_hard[perm]
        w_train = w_train[perm]

    def print_split_stats(split_name, labels, soft_labels, weights=None):
        counts = np.bincount(labels, minlength=len(EMOTION_COLS))
        soft_mass = soft_labels.sum(axis=0)
        distribution = ", ".join(
            f"{LABEL_NAMES[i]}={counts[i]:,} ({counts[i]/len(labels)*100:.1f}% | soft={soft_mass[i]:.1f})"
            for i in range(len(LABEL_NAMES))
        )
        w_info = f" | Trọng số mẫu TB w_i={weights.mean():.3f}" if weights is not None else ""
        print(f"{split_name}: {len(labels):,} ảnh{w_info}")
        print(f"  Phân phối: {distribution}")

    total_kept = len(y_train_hard) + len(y_val_hard) + len(y_test_hard)
    print("\n=== KẾT QUẢ KHỬ TRÙNG, LỌC NHIỄU VÀ CHIA TẬP STRATIFIED ===")
    print(f"Phương pháp chia tập: {split_method_str}")
    print(f"Bị loại do trùng MD5 pixel (Exact Deduplication): {dropped_md5_duplicates:,}")
    print(f"Bị loại do trùng thị giác dHash (Near-Duplicate): {dropped_phash_duplicates:,}")
    print(f"Bỏ qua do Usage không thuộc 3 tập: {ignored_usage:,}")
    print(f"Bị loại vì không còn phiếu cảm xúc > 1: {dropped_no_votes:,}")
    print(f"Bị loại vì unknown/NF >= cảm xúc cao nhất: {dropped_unknown_nf:,}")
    print(f"Bị loại vì hoà phiếu cao nhất (Ties): {dropped_ties:,}")
    print(f"Thu hồi vào tập Train (Rare-ties): {rescued_rare_ties:,}")
    print(f"Tổng ảnh sạch được giữ trong 3 tập: {total_kept:,}\n")

    if len(X_rare_ties) > 0:
        clean_counts = np.bincount(y_train_stratified_clean, minlength=len(EMOTION_COLS))
        clean_dist = ", ".join(
            f"{LABEL_NAMES[i]}={clean_counts[i]:,} ({clean_counts[i]/len(y_train_stratified_clean)*100:.1f}%)"
            for i in range(len(LABEL_NAMES))
        )
        print(f"Train (Chuẩn Stratified 70% trước khi cộng rare-ties): {len(y_train_stratified_clean):,} ảnh")
        print(f"  Phân phối chuẩn: {clean_dist}")

    print_split_stats("Train (70%)", y_train_hard, y_train_soft, w_train)
    print_split_stats("Validation (15%)", y_val_hard, y_val_soft, w_val)
    print_split_stats("Test (15%)", y_test_hard, y_test_soft, w_test)

    if return_soft_labels and return_sample_weights:
        return (
            (X_train, y_train_soft, y_train_hard, w_train),
            (X_val, y_val_soft, y_val_hard, w_val),
            (X_test, y_test_soft, y_test_hard, w_test),
        )

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
