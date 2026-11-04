import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from preprocessing import PREPROCESSING_CONFIG, preprocess_fer_gray_48x48

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
    Tính dHash 64-bit cho ảnh xám 2D.
    Trả về chuỗi hex 16 ký tự biểu diễn giá trị hash 64-bit.
    """
    resized = cv2.resize(img_2d, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    hash_val = 0
    for b in bits:
        hash_val = (hash_val << 1) | int(b)
    return f"{hash_val:016x}"


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """Tính khoảng cách Hamming giữa hai chuỗi hex dHash 64-bit."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def compute_file_sha256(filepath) -> str:
    """Tính mã băm SHA-256 của tệp tin trên đĩa."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_crowd_sample_weight(
    soft_probs,
    valid_votes_sum,
    unknown_votes=0,
    nf_votes=0,
    is_rescued_tie=False,
    min_weight=0.25,
):
    """
    Tính Trọng số Độ tin cậy của Mẫu (Sample Weight w_i in [min_weight, 1.0]) trực tiếp từ
    phân phối 10 phiếu bầu của FER+ (Barsoum et al., ICMI 2016) và nguyên lý Sample Weighting:
    - Thành phần 1 (Độ rõ ràng biểu cảm - Entropy): E_norm = -sum(p_c * log(p_c)) / log(8).
      Các ảnh đồng thuận cao (entropy thấp) nhận trọng số xấp xỉ 1.0; ảnh phân tán nhiều lớp nhận trọng số thấp hơn.
    - Thành phần 2 (Tỷ lệ phiếu hợp lệ so với phiếu nhiễu unknown/NF):
      valid_ratio = valid_votes_sum / (valid_votes_sum + unknown_votes + nf_votes).
    - Nếu là mẫu hoà phiếu được thu hồi riêng cho lớp hiếm (is_rescued_tie=True), nhân hệ số giảm 0.35
      để giảm ảnh hưởng nhiễu biên quyết định của lớp hiếm với lớp đa số.
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
    Xây dựng khóa phân tầng kết hợp (nhãn_chính, nhãn_phụ_lớn_nhất_có_xác_suất >= 0.15).
    LƯU Ý: Hàm này được giữ lại cho mục đích thử nghiệm tùy chọn và KHÔNG được sử dụng mặc định
    trong pipeline chính (mặc định pipeline dùng `use_multilabel_stratify=False`, phân tầng thuần
    theo một khóa duy nhất `y_hard` cho cả hai lần chia tập).
    """
    raw_keys = []
    for probs, primary in zip(y_soft, y_hard):
        sec_probs = probs.copy()
        sec_probs[primary] = 0.0
        sec_idx = int(np.argmax(sec_probs))
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

    final_series = pd.Series(final_keys)
    final_counts = final_series.value_counts()
    safe_keys = [
        k if final_counts[k] >= min_count_per_key else str(primary)
        for k, primary in zip(final_keys, y_hard)
    ]
    return np.asarray(safe_keys)


def _join_and_validate_sources(df_img: pd.DataFrame, df_label: pd.DataFrame, allow_positional_join: bool):
    """Ghép bảng ảnh và nhãn, kiểm tra tính toàn vẹn của Usage, Image name và gán source_row_index."""
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
        if set(df_label[id_col]) != set(df_img[id_col]):
            raise ValueError(f"Tập ID trong hai CSV không trùng khớp hoàn toàn ở cột '{id_col}'.")
        print(f"Đã ghép ảnh và nhãn bằng cột ID: {id_col}")
    else:
        if len(df_img) != len(df_label):
            raise ValueError(
                f"Không thể ghép dữ liệu: số dòng không khớp (ảnh={len(df_img)}, nhãn={len(df_label)})."
            )
        if not allow_positional_join:
            raise ValueError(
                "Hai CSV không có cột ID chung. Không thể xác minh ảnh-khớp-nhãn chỉ bằng số dòng. "
                "Hãy cung cấp khóa ID chung hoặc đặt allow_positional_join=True."
            )
        if "Usage" not in df_img.columns or "Usage" not in df_label.columns:
            raise ValueError("Cả hai file CSV phải có cột 'Usage' khi ghép theo vị trí dòng.")

        usage_img = df_img["Usage"].astype(str).str.strip().to_numpy()
        usage_lbl = df_label["Usage"].astype(str).str.strip().to_numpy()
        mismatch_indices = np.where(usage_img != usage_lbl)[0]
        if len(mismatch_indices) > 0:
            examples = [
                f"dòng {idx}: img='{usage_img[idx]}' != label='{usage_lbl[idx]}'"
                for idx in mismatch_indices[:5]
            ]
            raise ValueError(
                f"Phát hiện {len(mismatch_indices)} dòng có cột Usage lệch nhau giữa hai file CSV. "
                f"Ví dụ: {'; '.join(examples)}"
            )

        df = df_label.copy()
        df["pixels"] = df_img["pixels"].to_numpy()
        print("Ghép ảnh và nhãn theo thứ tự dòng (đã kiểm tra khớp 100% cột Usage tại mọi dòng).")

    # Tạo cột source_row_index ngay sau khi ghép (trước mọi bước loại)
    df["source_row_index"] = np.arange(len(df), dtype=np.int64)

    # Giữ cột "Image name" và kiểm tra tỷ lệ khớp chỉ số dòng nếu có dạng ferXXXXXXX.png
    if "Image name" in df.columns:
        df["Image name"] = df["Image name"].fillna("").astype(str).str.strip()
        pattern = re.compile(r"^fer(\d+)\.png$", re.IGNORECASE)
        matched_format = 0
        matched_index = 0
        for idx, name in zip(df["source_row_index"], df["Image name"]):
            m = pattern.match(name)
            if m:
                matched_format += 1
                if int(m.group(1)) == int(idx):
                    matched_index += 1
        if matched_format > 0:
            match_ratio = matched_index / matched_format * 100.0
            if matched_index < matched_format:
                warnings.warn(
                    f"Tỷ lệ khớp số trong 'Image name' (ferXXXXXXX.png) với chỉ số dòng: "
                    f"{matched_index}/{matched_format} ({match_ratio:.2f}%)."
                )
            else:
                print(
                    f"Kiểm tra 'Image name' (ferXXXXXXX.png): khớp {matched_index}/{matched_format} "
                    f"({match_ratio:.1f}%) với source_row_index."
                )
    else:
        df["Image name"] = ""

    return df


def resolve_reference_label(votes, source_row_index, policy="argmax", seed=42):
    """Reference label for stratification/metrics; soft targets remain unchanged.

    seeded breaks a top-vote tie reproducibly without preferring low class IDs.
    It is a declared reference convention, not a uniquely correct label for a tie.
    """
    tied = np.flatnonzero(votes == np.max(votes))
    if policy == "argmax" or len(tied) == 1:
        return int(tied[0])
    if policy != "seeded":
        raise ValueError(f"Unknown tie reference policy: {policy}")
    key = f"{int(seed)}:{int(source_row_index)}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return int(np.random.default_rng(local_seed).choice(tied))


def _evaluate_row_validity(
    row,
    has_unknown_col: bool,
    has_nf_col: bool,
    valid_usages: set,
    rare_set: set,
    drop_ties: bool,
    retain_rare_ties_in_train: bool,
    tie_reference_policy="argmax",
    tie_reference_seed=42,
):
    """
    Đánh giá tính hợp lệ của một dòng dữ liệu theo quy tắc lọc FER+:
    phiếu <= 1 đặt về 0; bỏ tổng phiếu = 0; bỏ max_vote <= unknown hoặc <= NF; bỏ ties khi drop_ties=True.
    Trả về dict chứa trạng thái hợp lệ, lý do loại (nếu có), max_vote sau lọc, nhãn cứng, nhãn mềm và trọng số.
    """
    usage = str(row["Usage"]).strip()
    votes = row[EMOTION_COLS].to_numpy(dtype=np.float32).copy()
    unknown_votes = int(row["unknown"]) if has_unknown_col else 0
    nf_votes = int(row["NF"]) if has_nf_col else 0

    votes[votes <= 1.0] = 0.0
    total_votes = float(votes.sum())
    max_vote = float(votes.max())
    raw_hard_label = resolve_reference_label(
        votes, row["source_row_index"], tie_reference_policy, tie_reference_seed
    )

    if usage not in valid_usages:
        return {
            "is_valid": False,
            "reject_reason": "ignored_usage",
            "max_vote": max_vote,
            "hard_label": raw_hard_label,
            "usage": usage,
        }

    if total_votes == 0.0:
        return {
            "is_valid": False,
            "reject_reason": "no_votes",
            "max_vote": 0.0,
            "hard_label": raw_hard_label,
            "usage": usage,
        }

    if max_vote <= unknown_votes or max_vote <= nf_votes:
        return {
            "is_valid": False,
            "reject_reason": "unknown_nf",
            "max_vote": max_vote,
            "hard_label": raw_hard_label,
            "usage": usage,
        }

    tied_classes = np.where(votes == max_vote)[0]
    is_tie = len(tied_classes) > 1
    tied_rare_classes = [int(c) for c in tied_classes if int(c) in rare_set]

    rescue_tie_for_train = (
        usage == "Training"
        and retain_rare_ties_in_train
        and len(tied_rare_classes) > 0
        and max_vote >= 3.0
    )
    if is_tie and drop_ties and not rescue_tie_for_train:
        return {
            "is_valid": False,
            "reject_reason": "ties",
            "max_vote": max_vote,
            "hard_label": raw_hard_label,
            "usage": usage,
        }

    soft_probs = votes / total_votes
    if is_tie and drop_ties and rescue_tie_for_train:
        hard_label = int(tied_rare_classes[-1])
        sample_w = compute_crowd_sample_weight(
            soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=True, min_weight=0.20
        )
        is_rescued = True
    else:
        hard_label = raw_hard_label
        sample_w = compute_crowd_sample_weight(
            soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=False
        )
        is_rescued = False

    return {
        "is_valid": True,
        "reject_reason": None,
        "max_vote": max_vote,
        "hard_label": hard_label,
        "soft_probs": soft_probs,
        "sample_w": sample_w,
        "is_rescued": is_rescued,
        "usage": usage,
    }


def load_fer2013_csv(
    img_csv_path,
    label_csv_path,
    return_soft_labels=False,
    return_sample_weights=False,
    use_stratify=True,
    use_multilabel_stratify=False,
    use_perceptual_dedup=True,
    near_dup_hamming=0,
    near_dup_pixel_mad_threshold=15.0,
    drop_ties=True,
    retain_rare_ties_in_train=False,
    rare_class_indices=(5, 6, 7),
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    split_manifest_dir=None,
    allow_positional_join=False,
    tie_reference_policy="argmax",
):
    """
    Đọc, làm sạch và chia tập bộ dữ liệu FER2013 kết hợp nhãn đám đông FER+ (Barsoum et al., ICMI 2016):
    1. Ghép ảnh và nhãn kèm kiểm tra đối chiếu cột Usage tại mọi dòng, bảo toàn `source_row_index` và `Image name`.
    2. Khử trùng lặp 2 tầng tất định:
       - Tầng 1 (SHA-256 pixel exact): chọn đại diện trong số các dòng HỢP LỆ (qua lọc usage, phiếu > 1,
         unknown/NF, ties) theo quy tắc `max_vote` cao nhất rồi `source_row_index` nhỏ nhất; nếu không có dòng
         nào hợp lệ thì cả nhóm bị loại.
       - Tầng 2 (cụm dHash 64-bit có xác nhận sai khác pixel gốc 48x48): chọn đại diện tất định tương tự.
    3. Chia tập bằng Stratified Split 2 bước theo tỷ lệ 70% Train / 15% Validation / 15% Test với khóa `y_hard`.
    """
    if not use_stratify:
        raise ValueError("Yêu cầu dự án bắt buộc stratify")

    if retain_rare_ties_in_train:
        raise ValueError("All filtering must finish BEFORE splitting; post-split tie rescue is disabled.")
    if tie_reference_policy not in {"argmax", "seeded"}:
        raise ValueError("tie_reference_policy must be argmax or seeded")

    df_img = pd.read_csv(img_csv_path)
    df_label = pd.read_csv(label_csv_path)
    initial_row_count = len(df_label)

    print("=== THỐNG KÊ DỮ LIỆU ĐẦU VÀO ===")
    print(f"Ảnh trong CSV ảnh: {len(df_img):,} dòng")
    print(f"Nhãn trong CSV nhãn: {len(df_label):,} dòng")

    df = _join_and_validate_sources(df_img, df_label, allow_positional_join=allow_positional_join)

    required = {"Usage", "pixels", *EMOTION_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    # --- ĐÁNH GIÁ TÍNH HỢP LỆ TỪNG DÒNG & TÍNH SHA-256 PIXEL ---
    df["image_id"] = df["pixels"].apply(
        lambda p: hashlib.sha256(" ".join(str(p).strip().split()).encode("utf-8")).hexdigest()
    )

    valid_usages = {"Training", "PublicTest", "PrivateTest"}
    rare_set = set(rare_class_indices) if rare_class_indices is not None else set()
    has_unknown_col = "unknown" in df.columns
    has_nf_col = "NF" in df.columns

    row_evals = [
        _evaluate_row_validity(
            row,
            has_unknown_col=has_unknown_col,
            has_nf_col=has_nf_col,
            valid_usages=valid_usages,
            rare_set=rare_set,
            drop_ties=drop_ties,
            retain_rare_ties_in_train=retain_rare_ties_in_train,
            tie_reference_policy=tie_reference_policy,
            tie_reference_seed=random_state,
        )
        for _, row in df.iterrows()
    ]

    df["_is_valid"] = [e["is_valid"] for e in row_evals]
    df["_reject_reason"] = [e["reject_reason"] for e in row_evals]
    df["_filtered_max_vote"] = [e["max_vote"] for e in row_evals]
    df["_eval_hard_label"] = [e["hard_label"] for e in row_evals]
    df["_eval_obj"] = row_evals

    # --- BƯỚC 1A: KHỬ TRÙNG PIXEL TUYỆT ĐỐI BẰNG SHA-256 (CHỌN ĐẠI DIỆN TRONG SỐ DÒNG HỢP LỆ) ---
    dedup_records = []
    exact_conflict_clusters = 0

    # Sắp xếp ưu tiên: dòng hợp lệ (_is_valid=True) đứng trước, sau đó max_vote giảm dần, source_row_index tăng dần
    df_sorted_exact = df.sort_values(
        by=["image_id", "_is_valid", "_filtered_max_vote", "source_row_index"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    dup_mask = df_sorted_exact.duplicated(subset=["image_id"], keep=False)
    if dup_mask.any():
        dup_groups = df_sorted_exact[dup_mask].groupby("image_id", sort=False)
        for _, grp in dup_groups:
            rep = grp.iloc[0]
            valid_sub = grp[grp["_is_valid"]]
            # Đếm xung đột nhãn cứng: ưu tiên xét trên các dòng hợp lệ trong nhóm (hoặc toàn nhóm nếu không dòng nào hợp lệ)
            check_sub = valid_sub if len(valid_sub) > 0 else grp
            if check_sub["_eval_hard_label"].nunique() > 1:
                exact_conflict_clusters += 1

            for _, dropped_row in grp.iloc[1:].iterrows():
                dedup_records.append({
                    "dropped_source_row_index": int(dropped_row["source_row_index"]),
                    "rep_source_row_index": int(rep["source_row_index"]),
                    "reason": "exact",
                    "hamming_distance": 0,
                    "pixel_mad": 0.0,
                    "dropped_hard_label": int(dropped_row["_eval_hard_label"]),
                    "rep_hard_label": int(rep["_eval_hard_label"]),
                    "dropped_usage": str(dropped_row["Usage"]).strip(),
                    "rep_usage": str(rep["Usage"]).strip(),
                })

    df_after_exact = (
        df_sorted_exact.drop_duplicates(subset=["image_id"], keep="first")
        .sort_values("source_row_index", kind="mergesort")
        .reset_index(drop=True)
    )
    dropped_sha256_duplicates = len(df) - len(df_after_exact)
    after_exact_count = len(df_after_exact)

    # --- BƯỚC 2: THU THẬP CÁC ĐẠI DIỆN HỢP LỆ & ĐẾM SỐ BỊ LOẠI THEO LÝ DO ---
    ignored_usage = 0
    dropped_no_votes = 0
    dropped_unknown_nf = 0
    dropped_ties = 0
    rescued_rare_ties = 0

    candidates = []
    for _, row in df_after_exact.iterrows():
        ev = row["_eval_obj"]
        if not ev["is_valid"]:
            reason = ev["reject_reason"]
            if reason == "ignored_usage":
                ignored_usage += 1
            elif reason == "no_votes":
                dropped_no_votes += 1
            elif reason == "unknown_nf":
                dropped_unknown_nf += 1
            elif reason == "ties":
                dropped_ties += 1
            continue

        pixels = np.fromstring(str(row["pixels"]), sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(f"Một ảnh có {pixels.size} pixel; cần đúng {48 * 48}.")
        raw_img = pixels.reshape(48, 48)

        candidates.append({
            "source_row_index": int(row["source_row_index"]),
            "image_id": str(row["image_id"]),
            "image_name": str(row["Image name"]),
            "usage": ev["usage"],
            "raw_img": raw_img,
            "max_vote": ev["max_vote"],
            "soft_probs": ev["soft_probs"],
            "hard_label": ev["hard_label"],
            "sample_w": ev["sample_w"],
            "is_rescued": ev["is_rescued"],
            "dhash": compute_dhash(raw_img) if use_perceptual_dedup else "",
        })

    after_vote_filter_count = len(candidates)

    # --- BƯỚC 1B: KHỬ TRÙNG DHASH + XÁC NHẬN PIXEL GỐC 48x48 VỚI ĐẠI DIỆN TẤT ĐỊNH ---
    dropped_dhash_duplicates = 0
    dhash_conflict_clusters = 0

    if use_perceptual_dedup and len(candidates) > 0:
        n_cand = len(candidates)
        parent = list(range(n_cand))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        hash_to_indices = {}
        for idx, c in enumerate(candidates):
            hash_to_indices.setdefault(c["dhash"], []).append(idx)

        unique_hashes = list(hash_to_indices.keys())
        candidate_pairs = []

        for h, idxs in hash_to_indices.items():
            if len(idxs) > 1:
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        candidate_pairs.append((idxs[i], idxs[j], 0))

        if near_dup_hamming > 0:
            for i in range(len(unique_hashes)):
                for j in range(i + 1, len(unique_hashes)):
                    h_dist = hamming_distance_hex(unique_hashes[i], unique_hashes[j])
                    if h_dist <= near_dup_hamming:
                        for idx_a in hash_to_indices[unique_hashes[i]]:
                            for idx_b in hash_to_indices[unique_hashes[j]]:
                                candidate_pairs.append((idx_a, idx_b, h_dist))

        for idx_a, idx_b, _ in candidate_pairs:
            img_a = candidates[idx_a]["raw_img"].astype(np.float32)
            img_b = candidates[idx_b]["raw_img"].astype(np.float32)
            mad = float(np.mean(np.abs(img_a - img_b)))
            if mad <= float(near_dup_pixel_mad_threshold):
                union(idx_a, idx_b)

        clusters = {}
        for idx in range(n_cand):
            clusters.setdefault(find(idx), []).append(idx)

        kept_candidates = []
        for comp_indices in clusters.values():
            if len(comp_indices) == 1:
                kept_candidates.append(candidates[comp_indices[0]])
            else:
                comp_sorted = sorted(
                    comp_indices,
                    key=lambda idx: (-candidates[idx]["max_vote"], candidates[idx]["source_row_index"]),
                )
                rep_cand = candidates[comp_sorted[0]]
                kept_candidates.append(rep_cand)

                cluster_labels = {candidates[idx]["hard_label"] for idx in comp_sorted}
                if len(cluster_labels) > 1:
                    dhash_conflict_clusters += 1

                for drop_idx in comp_sorted[1:]:
                    drop_cand = candidates[drop_idx]
                    h_dist = hamming_distance_hex(rep_cand["dhash"], drop_cand["dhash"])
                    mad = float(
                        np.mean(
                            np.abs(
                                rep_cand["raw_img"].astype(np.float32)
                                - drop_cand["raw_img"].astype(np.float32)
                            )
                        )
                    )
                    dedup_records.append({
                        "dropped_source_row_index": drop_cand["source_row_index"],
                        "rep_source_row_index": rep_cand["source_row_index"],
                        "reason": "dhash",
                        "hamming_distance": int(h_dist),
                        "pixel_mad": round(mad, 4),
                        "dropped_hard_label": drop_cand["hard_label"],
                        "rep_hard_label": rep_cand["hard_label"],
                        "dropped_usage": drop_cand["usage"],
                        "rep_usage": rep_cand["usage"],
                    })
                    dropped_dhash_duplicates += 1

        kept_candidates.sort(key=lambda c: c["source_row_index"])
    else:
        kept_candidates = candidates

    total_conflict_clusters = exact_conflict_clusters + dhash_conflict_clusters
    print(
        f"Số cụm trùng lặp có nhãn cứng xung đột: {total_conflict_clusters} "
        f"(exact={exact_conflict_clusters}, dhash={dhash_conflict_clusters})"
    )

    X_all, y_soft_all, y_hard_all, w_all = [], [], [], []
    usage_all, ids_all, src_idx_all, img_name_all = [], [], [], []

    X_rare_ties, y_soft_rare_ties, y_hard_rare_ties, w_rare_ties = [], [], [], []
    usage_rare_ties, ids_rare_ties, src_idx_rare_ties, img_name_rare_ties = [], [], [], []

    for cand in kept_candidates:
        norm_img = preprocess_fer_gray_48x48(cand["raw_img"])[..., np.newaxis]
        if cand["is_rescued"]:
            X_rare_ties.append(norm_img)
            y_soft_rare_ties.append(cand["soft_probs"])
            y_hard_rare_ties.append(cand["hard_label"])
            w_rare_ties.append(cand["sample_w"])
            usage_rare_ties.append(cand["usage"])
            ids_rare_ties.append(cand["image_id"])
            src_idx_rare_ties.append(cand["source_row_index"])
            img_name_rare_ties.append(cand["image_name"])
            rescued_rare_ties += 1
        else:
            X_all.append(norm_img)
            y_soft_all.append(cand["soft_probs"])
            y_hard_all.append(cand["hard_label"])
            w_all.append(cand["sample_w"])
            usage_all.append(cand["usage"])
            ids_all.append(cand["image_id"])
            src_idx_all.append(cand["source_row_index"])
            img_name_all.append(cand["image_name"])

    X_all = np.asarray(X_all, dtype=np.float32)
    y_soft_all = np.asarray(y_soft_all, dtype=np.float32)
    y_hard_all = np.asarray(y_hard_all, dtype=np.int64)
    w_all = np.asarray(w_all, dtype=np.float32)
    usage_all = np.asarray(usage_all)
    ids_all = np.asarray(ids_all)
    src_idx_all = np.asarray(src_idx_all, dtype=np.int64)
    img_name_all = np.asarray(img_name_all)

    # --- BƯỚC 3: CHIA TẬP STRATIFIED 70/15/15 THEO KHÓA DUY NHẤT Y_HARD ---
    holdout_size = val_size + test_size
    if not (0.0 < holdout_size < 1.0):
        raise ValueError(f"Tổng val_size + test_size phải nằm trong (0, 1), nhận được {holdout_size}.")

    num_classes = len(EMOTION_COLS)
    counts_before_split = np.bincount(y_hard_all, minlength=num_classes)
    if np.any(counts_before_split < 3):
        too_small = {LABEL_NAMES[i]: int(n) for i, n in enumerate(counts_before_split) if n < 3}
        raise ValueError(
            "Không đủ mẫu để chia train/val/test với stratify. "
            f"Lớp có dưới 3 mẫu: {too_small}"
        )

    if use_multilabel_stratify:
        stratify_keys_step1 = build_multilabel_stratify_keys(
            y_soft_all, y_hard_all, min_count_per_key=max(4, int(round(2.0 / min(val_size, test_size))))
        )
        strat_desc = "Multi-label Composite Stratification (Tùy chọn)"
    else:
        stratify_keys_step1 = y_hard_all
        strat_desc = "Stratify theo nhãn cảm xúc chính (y_hard)"

    # Lần chia 1: Train (70%) và Temp (30%)
    (
        X_train, X_temp,
        y_train_soft, y_temp_soft,
        y_train_hard, y_temp_hard,
        w_train, w_temp,
        ids_train, ids_temp,
        src_train, src_temp,
        name_train, name_temp,
        usage_train, usage_temp,
    ) = train_test_split(
        X_all, y_soft_all, y_hard_all, w_all,
        ids_all, src_idx_all, img_name_all, usage_all,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify_keys_step1,
    )

    train_counts_step1 = np.bincount(y_train_hard, minlength=num_classes)
    temp_counts_step1 = np.bincount(y_temp_hard, minlength=num_classes)
    if np.any(train_counts_step1 < 1) or np.any(temp_counts_step1 < 1):
        missing_info = {
            LABEL_NAMES[i]: {"train": int(train_counts_step1[i]), "temp": int(temp_counts_step1[i])}
            for i in range(num_classes)
            if train_counts_step1[i] < 1 or temp_counts_step1[i] < 1
        }
        raise ValueError(f"Sau lần chia 1, có lớp thiếu mẫu ở train hoặc temp: {missing_info}")

    if np.any(temp_counts_step1 < 2):
        too_small_temp = {LABEL_NAMES[i]: int(n) for i, n in enumerate(temp_counts_step1) if n < 2}
        raise ValueError(f"Tập temp có lớp dưới 2 mẫu nên không thể stratify val/test: {too_small_temp}")

    # Lần chia 2: Val (15%) và Test (15%) dùng khóa y_hard
    test_ratio_in_temp = test_size / holdout_size
    (
        X_val, X_test,
        y_val_soft, y_test_soft,
        y_val_hard, y_test_hard,
        w_val, w_test,
        ids_val, ids_test,
        src_val, src_test,
        name_val, name_test,
        usage_val, usage_test,
    ) = train_test_split(
        X_temp, y_temp_soft, y_temp_hard, w_temp,
        ids_temp, src_temp, name_temp, usage_temp,
        test_size=test_ratio_in_temp,
        random_state=random_state,
        stratify=y_temp_hard,
    )

    val_counts_step2 = np.bincount(y_val_hard, minlength=num_classes)
    test_counts_step2 = np.bincount(y_test_hard, minlength=num_classes)
    if np.any(val_counts_step2 < 1) or np.any(test_counts_step2 < 1):
        missing_info_2 = {
            LABEL_NAMES[i]: {"val": int(val_counts_step2[i]), "test": int(test_counts_step2[i])}
            for i in range(num_classes)
            if val_counts_step2[i] < 1 or test_counts_step2[i] < 1
        }
        raise ValueError(f"Sau lần chia 2, có lớp thiếu mẫu ở val hoặc test: {missing_info_2}")

    train_pct = int(round((1.0 - holdout_size) * 100))
    val_pct = int(round(val_size * 100))
    test_pct = int(round(test_size * 100))
    split_method_str = f"{strat_desc} ({train_pct}/{val_pct}/{test_pct}, random_state={random_state})"

    y_train_stratified_clean = y_train_hard.copy()

    # --- BƯỚC 4: KIỂM TRA HẬU ĐIỀU KIỆN (POST-CONDITIONS ASSERTS & BẢNG PHÂN PHỐI) ---
    set_ids_train, set_ids_val, set_ids_test = set(ids_train), set(ids_val), set(ids_test)
    assert set_ids_train.isdisjoint(set_ids_val), "Rò rỉ image_id giữa Train và Val!"
    assert set_ids_train.isdisjoint(set_ids_test), "Rò rỉ image_id giữa Train và Test!"
    assert set_ids_val.isdisjoint(set_ids_test), "Rò rỉ image_id giữa Val và Test!"

    set_src_train, set_src_val, set_src_test = set(src_train), set(src_val), set(src_test)
    assert set_src_train.isdisjoint(set_src_val), "Rò rỉ source_row_index giữa Train và Val!"
    assert set_src_train.isdisjoint(set_src_test), "Rò rỉ source_row_index giữa Train và Test!"
    assert set_src_val.isdisjoint(set_src_test), "Rò rỉ source_row_index giữa Val và Test!"

    total_valid_kept = len(kept_candidates)
    total_in_splits = len(X_train) + len(X_val) + len(X_test)
    assert total_in_splits == total_valid_kept, (
        f"Tổng mẫu 3 tập ({total_in_splits}) != tổng mẫu hợp lệ sau lọc ({total_valid_kept})!"
    )

    for s_name, s_X, s_y_soft, s_y_hard, s_w, s_ids in (
        ("Train", X_train, y_train_soft, y_train_hard, w_train, ids_train),
        ("Val", X_val, y_val_soft, y_val_hard, w_val, ids_val),
        ("Test", X_test, y_test_soft, y_test_hard, w_test, ids_test),
    ):
        n_s = len(s_X)
        assert len(s_y_soft) == n_s and len(s_y_hard) == n_s and len(s_w) == n_s and len(s_ids) == n_s, (
            f"Độ dài các mảng trong tập {s_name} không khớp nhau!"
        )
        c_s = np.bincount(s_y_hard, minlength=num_classes)
        assert np.all(c_s >= 1), f"Tập {s_name} có lớp không có mẫu nào: {c_s.tolist()}"

    actual_train_ratio = len(X_train) / total_in_splits * 100.0
    actual_val_ratio = len(X_val) / total_in_splits * 100.0
    actual_test_ratio = len(X_test) / total_in_splits * 100.0
    expected_train_ratio = (1.0 - holdout_size) * 100.0
    expected_val_ratio = val_size * 100.0
    expected_test_ratio = test_size * 100.0

    for split_name, actual, fraction in (
        ("train", len(X_train), 1.0 - holdout_size),
        ("val", len(X_val), val_size), ("test", len(X_test), test_size),
    ):
        if abs(actual - total_in_splits * fraction) > 1.0 + 1e-9:
            raise ValueError(f"{split_name}: split differs from requested ratio by more than one image")

    total_labels_combined = np.concatenate([y_train_hard, y_val_hard, y_test_hard])
    c_total = np.bincount(total_labels_combined, minlength=num_classes)
    c_train = np.bincount(y_train_hard, minlength=num_classes)
    c_val = np.bincount(y_val_hard, minlength=num_classes)
    c_test = np.bincount(y_test_hard, minlength=num_classes)

    pct_total = c_total / total_in_splits * 100.0
    pct_train = c_train / len(y_train_hard) * 100.0
    pct_val = c_val / len(y_val_hard) * 100.0
    pct_test = c_test / len(y_test_hard) * 100.0

    # Kiểm tra theo SỐ MẪU: abs(count_s[i] - total[i] * n_s / N) <= 2.0
    # Khi retain_rare_ties_in_train=True, bỏ qua tập Train và đối chiếu Val/Test với tập mẫu chuẩn trước khi cộng rare-ties
    if retain_rare_ties_in_train:
        N_ref = len(y_hard_all)
        total_ref = counts_before_split
        pct_ref = total_ref / N_ref * 100.0
        splits_to_check = [
            ("Val", c_val, pct_val, len(y_val_hard)),
            ("Test", c_test, pct_test, len(y_test_hard)),
        ]
    else:
        N_ref = total_in_splits
        total_ref = c_total
        pct_ref = pct_total
        splits_to_check = [
            ("Train", c_train, pct_train, len(y_train_hard)),
            ("Val", c_val, pct_val, len(y_val_hard)),
            ("Test", c_test, pct_test, len(y_test_hard)),
        ]

    max_class_dev = 0.0
    for s_name, count_s, pct_s, n_s in splits_to_check:
        for i in range(num_classes):
            expected_count = total_ref[i] * (n_s / N_ref)
            count_diff = abs(count_s[i] - expected_count)
            assert count_diff <= 2.0, (
                f"Số mẫu lớp {LABEL_NAMES[i]} trong tập {s_name} ({count_s[i]}) lệch quá 2.0 mẫu "
                f"so với kỳ vọng stratify ({expected_count:.2f})!"
            )
            dev_pct = abs(pct_s[i] - pct_ref[i])
            max_class_dev = max(max_class_dev, dev_pct)
            if n_s >= 2000 and dev_pct > 0.5:
                warnings.warn(
                    f"CẢNH BÁO: Tỷ lệ lớp {LABEL_NAMES[i]} trong tập {s_name} (n={n_s:,}) "
                    f"lệch {dev_pct:.3f} điểm % (> 0.5 điểm %) so với toàn bộ!"
                )

    # --- BƯỚC 5: LƯU MANIFEST, DEDUP REPORT VÀ SPLIT CONFIG JSON ---
    if split_manifest_dir is not None:
        manifest_dir = Path(split_manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        dedup_df = pd.DataFrame(
            dedup_records,
            columns=[
                "dropped_source_row_index",
                "rep_source_row_index",
                "reason",
                "hamming_distance",
                "pixel_mad",
                "dropped_hard_label",
                "rep_hard_label",
                "dropped_usage",
                "rep_usage",
            ],
        )
        dedup_df.to_csv(manifest_dir / "dedup_report.csv", index=False)

        split_dfs = []
        for split_name, s_ids, s_src, s_name, s_labels, s_soft, s_weights, s_usage in (
            ("train", ids_train, src_train, name_train, y_train_hard, y_train_soft, w_train, usage_train),
            ("val", ids_val, src_val, name_val, y_val_hard, y_val_soft, w_val, usage_val),
            ("test", ids_test, src_test, name_test, y_test_hard, y_test_soft, w_test, usage_test),
        ):
            m_df = pd.DataFrame({
                "image_id": s_ids,
                "source_row_index": s_src,
                "image_name": s_name,
                "label_id": s_labels,
                "label_name": [LABEL_NAMES[int(i)] for i in s_labels],
                "sample_weight": s_weights,
                "soft_label": [" ".join(f"{v:.6f}" for v in row) for row in s_soft],
                "usage_original": s_usage,
                "split": split_name,
            })
            m_df.to_csv(manifest_dir / f"{split_name}_ids.csv", index=False)
            split_dfs.append(m_df)

        all_splits_df = pd.concat(split_dfs, ignore_index=True)
        all_splits_df.to_csv(manifest_dir / "all_splits.csv", index=False)

        split_lookup = dict(zip(all_splits_df.source_row_index, all_splits_df.split))
        dedup_lookup = {int(r["dropped_source_row_index"]): r for r in dedup_records}
        audit_rows = []
        for _, row in df.iterrows():
            src = int(row["source_row_index"])
            ev = row["_eval_obj"]
            votes = row[EMOTION_COLS].to_numpy(dtype=np.float32).copy()
            votes[votes <= 1] = 0
            tied = np.flatnonzero(votes == votes.max()) if votes.sum() else np.array([], dtype=int)
            duplicate = dedup_lookup.get(src)
            status = "kept" if src in split_lookup else (
                "duplicate_" + duplicate["reason"] if duplicate else ev["reject_reason"] or "not_kept"
            )
            audit_rows.append(dict(source_row_index=src, image_id=row["image_id"],
                status=status, split=split_lookup.get(src, ""), is_tie=len(tied)>1,
                tied_classes=" ".join(map(str, tied)), reference_label=ev["hard_label"],
                representative_source_row_index=duplicate["rep_source_row_index"] if duplicate else None))
        audit_df = pd.DataFrame(audit_rows)
        audit_df.to_csv(manifest_dir / "filtering_report.csv", index=False)
        audit_df[audit_df.is_tie].to_csv(manifest_dir / "tie_analysis.csv", index=False)
        distribution = []
        for name, soft, hard in [("all_clean", y_soft_all, y_hard_all),
                ("train", y_train_soft, y_train_hard), ("val", y_val_soft, y_val_hard),
                ("test", y_test_soft, y_test_hard)]:
            for c, label in enumerate(LABEL_NAMES):
                distribution.append(dict(split=name, label=label, hard_count=int((hard==c).sum()),
                    soft_mass=float(soft[:,c].sum()), images_with_votes=int((soft[:,c]>0).sum())))
        pd.DataFrame(distribution).to_csv(manifest_dir / "class_distribution.csv", index=False)

        split_config = {
            "split_policy": "clean_all_then_stratified_holdout",
            "parameters": {
                "return_soft_labels": bool(return_soft_labels),
                "return_sample_weights": bool(return_sample_weights),
                "use_stratify": bool(use_stratify),
                "use_multilabel_stratify": bool(use_multilabel_stratify),
                "use_perceptual_dedup": bool(use_perceptual_dedup),
                "near_dup_hamming": int(near_dup_hamming),
                "near_dup_pixel_mad_threshold": float(near_dup_pixel_mad_threshold),
                "drop_ties": bool(drop_ties),
                "tie_reference_policy": tie_reference_policy,
                "tie_reference_seed": int(random_state),
                "retain_rare_ties_in_train": bool(retain_rare_ties_in_train),
                "rare_class_indices": list(rare_class_indices) if rare_class_indices is not None else [],
                "val_size": float(val_size),
                "test_size": float(test_size),
                "random_state": int(random_state),
                "allow_positional_join": bool(allow_positional_join),
            },
            "preprocessing_config": PREPROCESSING_CONFIG,
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
            },
            "source_files_sha256": {
                "img_csv": compute_file_sha256(img_csv_path),
                "label_csv": compute_file_sha256(label_csv_path),
            },
            "remaining_counts_after_each_step": {
                "initial_rows": int(initial_row_count),
                "after_exact_dedup": int(after_exact_count),
                "after_vote_and_tie_filtering": int(after_vote_filter_count),
                "after_dhash_dedup_final": int(total_valid_kept),
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "dropped_counts_by_reason": {
                "exact_sha256_duplicates": int(dropped_sha256_duplicates),
                "ignored_usage": int(ignored_usage),
                "no_valid_votes_gt_1": int(dropped_no_votes),
                "unknown_or_nf_dominant": int(dropped_unknown_nf),
                "ties_dropped": int(dropped_ties),
                "dhash_near_duplicates": int(dropped_dhash_duplicates),
                "rescued_rare_ties": int(rescued_rare_ties),
            },
            "conflict_clusters": {
                "exact_conflict_clusters": int(exact_conflict_clusters),
                "dhash_conflict_clusters": int(dhash_conflict_clusters),
                "total_conflict_clusters": int(total_conflict_clusters),
            },
            "train_class_counts": {
                LABEL_NAMES[i]: int(c_train[i]) for i in range(num_classes)
            },
        }
        with open(manifest_dir / "split_config.json", "w", encoding="utf-8") as f:
            json.dump(split_config, f, indent=2, ensure_ascii=False)

        print(f"Đã lưu manifest, dedup_report.csv và split_config.json tại: {manifest_dir.resolve()}")

    print("\n=== KẾT QUẢ KHỬ TRÙNG, LỌC NHIỄU VÀ CHIA TẬP STRATIFIED ===")
    print(f"Phương pháp chia tập: {split_method_str}")
    print(f"Bị loại do trùng pixel SHA-256 (Exact Deduplication): {dropped_sha256_duplicates:,}")
    print(
        f"Bị loại do dHash (hamming<={near_dup_hamming}, pixel MAD<={near_dup_pixel_mad_threshold}): "
        f"{dropped_dhash_duplicates:,}"
    )
    print(f"Bỏ qua do Usage không thuộc 3 tập: {ignored_usage:,}")
    print(f"Bị loại vì không còn phiếu cảm xúc > 1: {dropped_no_votes:,}")
    print(f"Bị loại vì unknown/NF >= cảm xúc cao nhất: {dropped_unknown_nf:,}")
    print(f"Bị loại vì hoà phiếu cao nhất (Ties): {dropped_ties:,}")
    print(f"Thu hồi vào tập Train (Rare-ties): {rescued_rare_ties:,}")
    print(f"Tổng ảnh hợp lệ được giữ trong 3 tập: {total_in_splits:,}")
    print(
        f"[KIỂM TRA HẬU ĐIỀU KIỆN ĐẠT] 3 tập rời nhau hoàn toàn | "
        f"Tỷ lệ tập thực tế: {actual_train_ratio:.2f}% / {actual_val_ratio:.2f}% / {actual_test_ratio:.2f}% | "
        f"Độ lệch tỷ lệ lớp tối đa: {max_class_dev:.3f} điểm %\n"
    )

    print("=== BẢNG PHÂN PHỐI SỐ LƯỢNG VÀ TỶ LỆ (%) THEO LỚP ===")
    header = f"{'Lớp cảm xúc':<12} | {'Train':>14} | {'Val':>14} | {'Test':>14} | {'Tổng':>14}"
    print(header)
    print("-" * len(header))
    for i in range(num_classes):
        print(
            f"{LABEL_NAMES[i]:<12} | "
            f"{c_train[i]:>6,} ({pct_train[i]:>4.1f}%) | "
            f"{c_val[i]:>6,} ({pct_val[i]:>4.1f}%) | "
            f"{c_test[i]:>6,} ({pct_test[i]:>4.1f}%) | "
            f"{c_total[i]:>6,} ({pct_total[i]:>4.1f}%)"
        )
    print("-" * len(header))
    print(
        f"{'TỔNG CỘNG':<12} | "
        f"{len(X_train):>6,} ({actual_train_ratio:>4.1f}%) | "
        f"{len(X_val):>6,} ({actual_val_ratio:>4.1f}%) | "
        f"{len(X_test):>6,} ({actual_test_ratio:>4.1f}%) | "
        f"{total_in_splits:>6,} (100.0%)\n"
    )

    if len(X_rare_ties) > 0:
        clean_counts = np.bincount(y_train_stratified_clean, minlength=num_classes)
        clean_dist = ", ".join(
            f"{LABEL_NAMES[i]}={clean_counts[i]:,} ({clean_counts[i]/len(y_train_stratified_clean)*100:.1f}%)"
            for i in range(num_classes)
        )
        print(f"Train (Chuẩn Stratified trước khi cộng rare-ties): {len(y_train_stratified_clean):,} ảnh")
        print(f"  Phân phối chuẩn: {clean_dist}")

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


def load_split_from_manifest(
    img_csv,
    label_csv,
    manifest_dir,
    allow_positional_join=True,
    verify_source_hash=True,
):
    """
    Dựng lại chính xác các mảng (X, y_soft, y_hard, w) cho 3 tập (train, val, test)
    trực tiếp từ thư mục manifest đã lưu mà KHÔNG chia lại tập.
    - Khi `verify_source_hash=True`, đối chiếu SHA-256 của `img_csv` và `label_csv` với
      `split_config.json["source_files_sha256"]`; nếu lệch thì raise ValueError.
    - Kiểm tra đối chiếu (assert) image_id, nhãn cứng, nhãn mềm và trọng số mẫu khớp với manifest.
    """
    manifest_path = Path(manifest_dir)
    config_path = manifest_path / "split_config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        params = cfg.get("parameters", {})
        tie_reference_policy = params.get("tie_reference_policy", "argmax")
        tie_reference_seed = params.get("tie_reference_seed", params.get("random_state", 42))
        drop_ties = bool(params.get("drop_ties", True))
        retain_rare_ties = bool(params.get("retain_rare_ties_in_train", False))
        rare_set = set(params.get("rare_class_indices", [5, 6, 7]))

        if verify_source_hash:
            expected_hashes = cfg.get("source_files_sha256", {})
            expected_img_hash = expected_hashes.get("img_csv")
            expected_lbl_hash = expected_hashes.get("label_csv")
            actual_img_hash = compute_file_sha256(img_csv)
            actual_lbl_hash = compute_file_sha256(label_csv)

            if expected_img_hash and actual_img_hash != expected_img_hash:
                raise ValueError(
                    f"SHA-256 của img_csv ({actual_img_hash}) không khớp với split_config.json ({expected_img_hash})."
                )
            if expected_lbl_hash and actual_lbl_hash != expected_lbl_hash:
                raise ValueError(
                    f"SHA-256 của label_csv ({actual_lbl_hash}) không khớp với split_config.json ({expected_lbl_hash})."
                )
    else:
        if verify_source_hash:
            raise FileNotFoundError(
                f"Không tìm thấy {config_path} để xác minh SHA-256 của file nguồn (verify_source_hash=True)."
            )
        tie_reference_policy = "argmax"
        tie_reference_seed = 42
        drop_ties = True
        retain_rare_ties = False
        rare_set = {5, 6, 7}

    df_img = pd.read_csv(img_csv)
    df_label = pd.read_csv(label_csv)

    df = _join_and_validate_sources(df_img, df_label, allow_positional_join=allow_positional_join)
    df["image_id"] = df["pixels"].apply(
        lambda p: hashlib.sha256(" ".join(str(p).strip().split()).encode("utf-8")).hexdigest()
    )

    df_indexed = df.set_index("source_row_index", drop=False)

    reconstructed = []
    for split_name in ("train", "val", "test"):
        split_csv = manifest_path / f"{split_name}_ids.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy file manifest: {split_csv}")

        m_df = pd.read_csv(split_csv)
        X_list, y_soft_list, y_hard_list, w_list = [], [], [], []

        for _, m_row in m_df.iterrows():
            src_idx = int(m_row["source_row_index"])
            if src_idx not in df_indexed.index:
                raise KeyError(f"Không tìm thấy source_row_index={src_idx} trong CSV nguồn.")

            src_row = df_indexed.loc[src_idx]
            assert str(src_row["image_id"]) == str(m_row["image_id"]), (
                f"Lệch image_id tại source_row_index={src_idx}: "
                f"CSV='{src_row['image_id']}' != Manifest='{m_row['image_id']}'"
            )

            votes = src_row[EMOTION_COLS].to_numpy(dtype=np.float32).copy()
            unknown_votes = int(src_row["unknown"]) if "unknown" in df.columns else 0
            nf_votes = int(src_row["NF"]) if "NF" in df.columns else 0

            votes[votes <= 1.0] = 0.0
            total_votes = float(votes.sum())
            assert total_votes > 0.0, f"Dòng {src_idx} trong manifest có tổng phiếu hợp lệ bằng 0!"

            max_vote = float(votes.max())
            tied_classes = np.where(votes == max_vote)[0]
            is_tie = len(tied_classes) > 1
            tied_rare_classes = [int(c) for c in tied_classes if int(c) in rare_set]
            usage = str(src_row["Usage"]).strip()

            rescue_tie_for_train = (
                usage == "Training"
                and retain_rare_ties
                and len(tied_rare_classes) > 0
                and max_vote >= 3.0
            )

            soft_probs = votes / total_votes
            if is_tie and drop_ties and rescue_tie_for_train:
                hard_label = int(tied_rare_classes[-1])
                sample_w = compute_crowd_sample_weight(
                    soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=True, min_weight=0.20
                )
            else:
                hard_label = resolve_reference_label(
                    votes, src_idx, tie_reference_policy, tie_reference_seed
                )
                sample_w = compute_crowd_sample_weight(
                    soft_probs, total_votes, unknown_votes, nf_votes, is_rescued_tie=False
                )

            manifest_label = int(m_row["label_id"])
            manifest_weight = float(m_row["sample_weight"])
            manifest_soft = np.fromstring(str(m_row["soft_label"]), sep=" ", dtype=np.float32)

            assert hard_label == manifest_label, (
                f"Nhãn cứng tính lại ({hard_label}) != manifest ({manifest_label}) tại dòng {src_idx}!"
            )
            assert np.isclose(sample_w, manifest_weight, atol=1e-5), (
                f"Trọng số mẫu tính lại ({sample_w}) != manifest ({manifest_weight}) tại dòng {src_idx}!"
            )
            assert np.allclose(soft_probs, manifest_soft, atol=1e-5), (
                f"Nhãn mềm tính lại != manifest tại dòng {src_idx}!"
            )

            pixels = np.fromstring(str(src_row["pixels"]), sep=" ", dtype=np.uint8).reshape(48, 48)
            norm_img = preprocess_fer_gray_48x48(pixels)[..., np.newaxis]

            X_list.append(norm_img)
            y_soft_list.append(soft_probs)
            y_hard_list.append(hard_label)
            w_list.append(sample_w)

        reconstructed.append((
            np.asarray(X_list, dtype=np.float32),
            np.asarray(y_soft_list, dtype=np.float32),
            np.asarray(y_hard_list, dtype=np.int64),
            np.asarray(w_list, dtype=np.float32),
        ))

    return tuple(reconstructed)

def extend_train_with_ties(*args, **kwargs):
    """Retired: append-after-split violates the required 70/15/15 protocol."""
    raise RuntimeError("Use load_fer2013_csv(drop_ties=False): clean all data, then stratify 70/15/15.")
