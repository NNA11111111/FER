# FER — Tối ưu và Dự phòng

| Folder | Nội dung |
|---|---|
| [TỐI ƯU](TỐI%20ƯU/) | Bản tối ưu được giữ nguyên từ nhánh `toi-uu`. |
| [DỰ PHÒNG](DỰ%20PHÒNG/) | Code mới chạy hai nhánh giữ/bỏ hòa phiếu, chia stratify 70–15–15 riêng và McNemar trên giao hai test. |

## Notebook dự phòng

1. Làm sạch và chia riêng hai bộ dữ liệu.
2. Định nghĩa các hàm huấn luyện.
3. Train lần lượt hai nhánh, mỗi nhánh ba seed; chọn mô hình bằng validation riêng.
4. Xuất kết quả test riêng và Grad-CAM có nhãn thật/dự đoán.
5. Ghép ảnh chung giữa hai test để tính McNemar, xuất độ bao phủ và phân bố lớp.

**Các CSV, checkpoint và ảnh kết quả đang có trong folder DỰ PHÒNG là kết quả của lần train trước, không phải kết quả của code mới.**
Notebook mới ghi kết quả vào `two_branches_stratified_70_15_15`, tách riêng `drop_ties`, `keep_ties` và `test_intersection`.
Phần giao test không bảo đảm giữ tỷ lệ lớp và chỉ gồm ảnh không hòa phiếu. McNemar không kiểm định Macro-F1.

Lần cập nhật này chỉ sửa và tổ chức mã nguồn; chưa chạy notebook, train hay test.
