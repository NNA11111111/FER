import cv2
import numpy as np
import pandas as pd
import pytest

from data_utils import (
    EMOTION_COLS,
    LABEL_NAMES,
    load_fer2013_csv,
    load_split_from_manifest,
)
from preprocessing import preprocess_fer_gray_48x48


def _legacy_preprocess_code(raw_img_uint8: np.ndarray) -> np.ndarray:
    """Đoạn code tiền xử lý cũ trước khi tách ra preprocessing.py."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = clahe.apply(raw_img_uint8)
    image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
    return image


def _create_synthetic_fer_csvs(tmp_path, n_base=400, seed=123):
    """
    Tạo 2 file CSV tổng hợp (~420 dòng) gồm:
    - 400 ảnh gốc thuộc 8 lớp mất cân bằng (lớp 0,1 nhiều; lớp 5,6,7 hiếm).
    - 10 ảnh trùng hệt pixel (SHA-256) nhưng có Usage và số phiếu hợp lệ khác nhau (xung đột nhãn cứng).
    - 10 ảnh gần giống (cùng cấu trúc dHash, sai khác pixel nhỏ MAD <= 15.0).
    """
    rng = np.random.RandomState(seed)

    class_probs = [0.32, 0.25, 0.12, 0.12, 0.09, 0.03, 0.04, 0.03]
    labels = rng.choice(8, size=n_base, p=class_probs)
    usages_pool = ["Training"] * 280 + ["PublicTest"] * 60 + ["PrivateTest"] * 60

    img_rows = []
    lbl_rows = []
    base_images = []

    for i in range(n_base):
        base_grid = rng.randint(20, 230, size=(48, 48), dtype=np.uint8)
        base_images.append(base_grid)
        pixel_str = " ".join(map(str, base_grid.flatten().tolist()))

        cls_id = int(labels[i])
        votes = np.zeros(8, dtype=int)
        votes[cls_id] = rng.randint(6, 10)
        sec_cls = (cls_id + 1) % 8
        votes[sec_cls] = 10 - votes[cls_id]

        usage = usages_pool[i]
        img_rows.append({"Usage": usage, "pixels": pixel_str})
        lbl_rows.append({
            "Usage": usage,
            "Image name": f"fer{i:07d}.png",
            **{EMOTION_COLS[c]: int(votes[c]) for c in range(8)},
            "unknown": 0,
            "NF": 0,
        })

    # Thêm 10 ảnh trùng hệt pixel hợp lệ nhưng có max_vote=10 cao hơn bản gốc (6..9)
    for k in range(10):
        idx_ref = k
        pixel_str = img_rows[idx_ref]["pixels"]
        row_idx = len(img_rows)
        alt_usage = "PrivateTest" if img_rows[idx_ref]["Usage"] == "Training" else "Training"

        votes = np.zeros(8, dtype=int)
        conflict_cls = (int(labels[idx_ref]) + 2) % 8
        votes[conflict_cls] = 10

        img_rows.append({"Usage": alt_usage, "pixels": pixel_str})
        lbl_rows.append({
            "Usage": alt_usage,
            "Image name": f"fer{row_idx:07d}.png",
            **{EMOTION_COLS[c]: int(votes[c]) for c in range(8)},
            "unknown": 0,
            "NF": 0,
        })

    # Thêm 10 ảnh biến đổi nhẹ trên 1 pixel (cùng dHash, MAD ~ 0.002 <= 15.0)
    for k in range(10, 20):
        arr = base_images[k].copy()
        arr[0, 0] = np.uint8((int(arr[0, 0]) + 1) % 256)
        pixel_str = " ".join(map(str, arr.flatten().tolist()))
        row_idx = len(img_rows)

        votes = np.zeros(8, dtype=int)
        votes[int(labels[k])] = 5

        img_rows.append({"Usage": "Training", "pixels": pixel_str})
        lbl_rows.append({
            "Usage": "Training",
            "Image name": f"fer{row_idx:07d}.png",
            **{EMOTION_COLS[c]: int(votes[c]) for c in range(8)},
            "unknown": 0,
            "NF": 0,
        })

    img_csv = tmp_path / "fer2013_synth.csv"
    lbl_csv = tmp_path / "fer2013new_synth.csv"
    pd.DataFrame(img_rows).to_csv(img_csv, index=False)
    pd.DataFrame(lbl_rows).to_csv(lbl_csv, index=False)
    return img_csv, lbl_csv


def test_shared_preprocessing_matches_legacy_100_samples():
    """B1: Kiểm tra với 100 ảnh mẫu, hàm tiền xử lý chung khớp tuyệt đối (np.array_equal) với code cũ."""
    rng = np.random.RandomState(2026)
    for _ in range(100):
        sample_img = rng.randint(0, 256, size=(48, 48), dtype=np.uint8)
        expected = _legacy_preprocess_code(sample_img)
        actual = preprocess_fer_gray_48x48(sample_img)
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected), "Kết quả hàm tiền xử lý mới bị lệch so với code cũ!"


def test_detect_usage_mismatch_raises_value_error(tmp_path):
    """A1: Phát hiện cột Usage lệch nhau giữa 2 file khi allow_positional_join=True."""
    img_csv, lbl_csv = _create_synthetic_fer_csvs(tmp_path)
    df_lbl = pd.read_csv(lbl_csv)
    df_lbl.loc[15, "Usage"] = "PublicTest" if df_lbl.loc[15, "Usage"] == "Training" else "Training"
    corrupted_lbl_csv = tmp_path / "fer2013new_corrupted.csv"
    df_lbl.to_csv(corrupted_lbl_csv, index=False)

    with pytest.raises(ValueError, match="Usage lệch nhau"):
        load_fer2013_csv(
            img_csv,
            corrupted_lbl_csv,
            allow_positional_join=True,
        )


def test_use_stratify_false_raises_value_error(tmp_path):
    """A3: Nhánh use_stratify=False bắt buộc raise ValueError theo yêu cầu dự án."""
    img_csv, lbl_csv = _create_synthetic_fer_csvs(tmp_path)
    with pytest.raises(ValueError, match="Yêu cầu dự án bắt buộc stratify"):
        load_fer2013_csv(
            img_csv,
            lbl_csv,
            use_stratify=False,
            allow_positional_join=True,
        )


def test_pipeline_disjoint_stratified_reproducible_and_manifest(tmp_path):
    """
    Kiểm tra toàn diện A2, A3, A4 (theo số mẫu lệch <= 2.0), A5:
    - Khử trùng exact + dHash không raise lỗi khi xung đột metadata, xuất dedup_report.csv.
    - ID và source_row_index của 3 tập đôi một rời nhau.
    - Stratify giữ số lượng mẫu mỗi lớp sát kỳ vọng (abs(count_s[i] - total[i] * n_s / N) <= 2.0).
    - Chạy hai lần cho kết quả giống hệt 100%.
    - load_split_from_manifest tái tạo đúng hoàn toàn các mảng (X, y_soft, y_hard, w).
    """
    img_csv, lbl_csv = _create_synthetic_fer_csvs(tmp_path)
    manifest_dir_1 = tmp_path / "manifest_run1"
    manifest_dir_2 = tmp_path / "manifest_run2"

    run1 = load_fer2013_csv(
        img_csv,
        lbl_csv,
        return_soft_labels=True,
        return_sample_weights=True,
        use_stratify=True,
        use_multilabel_stratify=False,
        use_perceptual_dedup=True,
        near_dup_hamming=0,
        near_dup_pixel_mad_threshold=15.0,
        drop_ties=True,
        split_manifest_dir=manifest_dir_1,
        allow_positional_join=True,
    )

    run2 = load_fer2013_csv(
        img_csv,
        lbl_csv,
        return_soft_labels=True,
        return_sample_weights=True,
        use_stratify=True,
        use_multilabel_stratify=False,
        use_perceptual_dedup=True,
        near_dup_hamming=0,
        near_dup_pixel_mad_threshold=15.0,
        drop_ties=True,
        split_manifest_dir=manifest_dir_2,
        allow_positional_join=True,
    )

    # 1. Chạy hai lần cho kết quả giống hệt nhau
    for split_idx in range(3):
        for arr_idx in range(4):
            assert np.array_equal(run1[split_idx][arr_idx], run2[split_idx][arr_idx])

    # 2. Kiểm tra độ lệch số mẫu từng lớp <= 2.0 theo Mục 1
    y_tr, y_va, y_te = run1[0][2], run1[1][2], run1[2][2]
    N_total = len(y_tr) + len(y_va) + len(y_te)
    c_total = np.bincount(np.concatenate([y_tr, y_va, y_te]), minlength=len(LABEL_NAMES))
    for y_s in (y_tr, y_va, y_te):
        n_s = len(y_s)
        c_s = np.bincount(y_s, minlength=len(LABEL_NAMES))
        for i in range(len(LABEL_NAMES)):
            assert abs(c_s[i] - c_total[i] * n_s / N_total) <= 2.0

    # 3. Kiểm tra các file manifest và dedup_report.csv
    assert (manifest_dir_1 / "dedup_report.csv").exists()
    assert (manifest_dir_1 / "all_splits.csv").exists()
    assert (manifest_dir_1 / "split_config.json").exists()

    dedup_df = pd.read_csv(manifest_dir_1 / "dedup_report.csv")
    assert len(dedup_df) >= 20
    assert set(dedup_df["reason"]).issuperset({"exact", "dhash"})

    exact_rows = dedup_df[dedup_df["reason"] == "exact"]
    assert all(exact_rows["rep_source_row_index"] >= 400)

    # 4. Kiểm tra ID và source_row_index 3 tập rời nhau
    tr_df = pd.read_csv(manifest_dir_1 / "train_ids.csv")
    va_df = pd.read_csv(manifest_dir_1 / "val_ids.csv")
    te_df = pd.read_csv(manifest_dir_1 / "test_ids.csv")

    assert set(tr_df["image_id"]).isdisjoint(set(va_df["image_id"]))
    assert set(tr_df["image_id"]).isdisjoint(set(te_df["image_id"]))
    assert set(va_df["image_id"]).isdisjoint(set(te_df["image_id"]))

    assert set(tr_df["source_row_index"]).isdisjoint(set(va_df["source_row_index"]))
    assert set(tr_df["source_row_index"]).isdisjoint(set(te_df["source_row_index"]))
    assert set(va_df["source_row_index"]).isdisjoint(set(te_df["source_row_index"]))

    # 5. Kiểm tra load_split_from_manifest tái tạo khớp 100%
    reconstructed = load_split_from_manifest(
        img_csv,
        lbl_csv,
        manifest_dir=manifest_dir_1,
        allow_positional_join=True,
        verify_source_hash=True,
    )
    for split_idx in range(3):
        assert np.array_equal(run1[split_idx][0], reconstructed[split_idx][0])
        assert np.allclose(run1[split_idx][1], reconstructed[split_idx][1], atol=1e-6)
        assert np.array_equal(run1[split_idx][2], reconstructed[split_idx][2])
        assert np.allclose(run1[split_idx][3], reconstructed[split_idx][3], atol=1e-6)


def test_exact_dedup_rescues_valid_duplicate_when_higher_vote_is_tie(tmp_path):
    """
    Mục 6a (và Mục 2): Khi một nhóm trùng pixel có 1 bản có max_vote cao hơn (ví dụ 5-5) nhưng bị loại
    do hòa phiếu (drop_ties=True), và 1 bản còn lại hợp lệ (ví dụ phiếu 4-3, max_vote=4),
    thì bản hợp lệ phải được chọn làm đại diện và giữ lại trong tập chia.
    """
    img_csv, lbl_csv = _create_synthetic_fer_csvs(tmp_path)
    df_img = pd.read_csv(img_csv)
    df_lbl = pd.read_csv(lbl_csv)

    # Tạo 1 ảnh mới hoàn toàn và thêm 2 dòng trùng hệt pixel vào cuối bảng
    rng = np.random.RandomState(999)
    special_img = rng.randint(10, 240, size=(48, 48), dtype=np.uint8)
    special_pixels = " ".join(map(str, special_img.flatten().tolist()))

    tie_row_idx = len(df_img)      # Dòng có max_vote = 5 nhưng bị tie (5 Neutral vs 5 Happiness)
    valid_row_idx = len(df_img) + 1  # Dòng hợp lệ có max_vote = 4 (4 Surprise vs 3 Sadness)

    df_img = pd.concat([
        df_img,
        pd.DataFrame([
            {"Usage": "Training", "pixels": special_pixels},
            {"Usage": "Training", "pixels": special_pixels},
        ]),
    ], ignore_index=True)

    tie_lbl = {c: 0 for c in EMOTION_COLS}
    tie_lbl["neutral"] = 5
    tie_lbl["happiness"] = 5

    valid_lbl = {c: 0 for c in EMOTION_COLS}
    valid_lbl["surprise"] = 4
    valid_lbl["sadness"] = 3

    df_lbl = pd.concat([
        df_lbl,
        pd.DataFrame([
            {"Usage": "Training", "Image name": f"fer{tie_row_idx:07d}.png", **tie_lbl, "unknown": 0, "NF": 0},
            {"Usage": "Training", "Image name": f"fer{valid_row_idx:07d}.png", **valid_lbl, "unknown": 0, "NF": 0},
        ]),
    ], ignore_index=True)

    img_special_csv = tmp_path / "img_tie_dup.csv"
    lbl_special_csv = tmp_path / "lbl_tie_dup.csv"
    df_img.to_csv(img_special_csv, index=False)
    df_lbl.to_csv(lbl_special_csv, index=False)

    manifest_dir = tmp_path / "manifest_tie_dup"
    load_fer2013_csv(
        img_special_csv,
        lbl_special_csv,
        return_soft_labels=True,
        return_sample_weights=True,
        drop_ties=True,
        split_manifest_dir=manifest_dir,
        allow_positional_join=True,
    )

    all_splits = pd.read_csv(manifest_dir / "all_splits.csv")
    dedup_report = pd.read_csv(manifest_dir / "dedup_report.csv")

    # Bản hợp lệ (valid_row_idx) phải được giữ lại trong manifest 3 tập
    assert valid_row_idx in set(all_splits["source_row_index"])
    assert tie_row_idx not in set(all_splits["source_row_index"])

    # Trong báo cáo khử trùng, dòng tie_row_idx bị loại bởi đại diện valid_row_idx với lý do 'exact'
    match_dedup = dedup_report[
        (dedup_report["dropped_source_row_index"] == tie_row_idx)
        & (dedup_report["rep_source_row_index"] == valid_row_idx)
        & (dedup_report["reason"] == "exact")
    ]
    assert len(match_dedup) == 1


def test_load_split_from_manifest_raises_on_modified_source_csv(tmp_path):
    """
    Mục 6b (và Mục 3): load_split_from_manifest phải raise ValueError khi nội dung
    của img_csv hoặc label_csv bị thay đổi so với mã băm SHA-256 trong split_config.json.
    """
    img_csv, lbl_csv = _create_synthetic_fer_csvs(tmp_path)
    manifest_dir = tmp_path / "manifest_hash_check"

    load_fer2013_csv(
        img_csv,
        lbl_csv,
        return_soft_labels=True,
        return_sample_weights=True,
        split_manifest_dir=manifest_dir,
        allow_positional_join=True,
    )

    # Sửa đổi nhẹ 1 file CSV nguồn (thêm khoảng trắng hoặc đổi 1 giá trị ở dòng không dùng)
    tampered_lbl_csv = tmp_path / "lbl_tampered.csv"
    df_lbl = pd.read_csv(lbl_csv)
    df_lbl.loc[0, "unknown"] = int(df_lbl.loc[0, "unknown"]) + 1
    df_lbl.to_csv(tampered_lbl_csv, index=False)

    with pytest.raises(ValueError, match="SHA-256"):
        load_split_from_manifest(
            img_csv,
            tampered_lbl_csv,
            manifest_dir=manifest_dir,
            allow_positional_join=True,
            verify_source_hash=True,
        )