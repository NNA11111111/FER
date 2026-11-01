import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FER2013Dataset(Dataset):
    """
    Dataset FER2013 / FER+ với thiết kế Augmentation bảo toàn vi biểu cảm (Micro-expression Safe)
    kết hợp bảo vệ đặc biệt cho 3 lớp hiếm (Disgust=5, Fear=6, Contempt=7):
    - Co giãn nhẹ ±8%, Dịch ±8%, Xoay ±10°, Lật ngang 50%.
    - TẮT RandomErasing đối với các mẫu thuộc 3 lớp hiếm (Disgust, Fear, Contempt) ngay trong __getitem__,
      ngăn chặn 100% nguy cơ vùng xóa ngẫu nhiên che mất khoé miệng nhếch lệch của Contempt (AU12/AU14),
      sống mũi nhăn của Disgust (AU9) hoặc vùng chân mày căng của Fear (AU1+2+4).
    - Hỗ trợ bật/tắt RandomErasing động qua set_erasing_enabled() giữa các giai đoạn huấn luyện.
    """
    def __init__(
        self,
        X,
        y,
        is_train=False,
        erasing_prob=0.15,
        erasing_scale=(0.01, 0.04),
        sample_weights=None,
        rare_classes=(5, 6, 7),
    ):
        self.X = X
        self.y = y
        self.is_train = is_train
        self.erasing_enabled = is_train
        self.rare_classes = set(rare_classes) if rare_classes is not None else set()
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else None
        )

        if self.is_train:
            self.spatial_transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            self.erasing_transform = (
                transforms.RandomErasing(p=erasing_prob, scale=erasing_scale, value="random")
                if erasing_prob > 0.0
                else None
            )
        else:
            self.spatial_transform = None
            self.erasing_transform = None

    def set_erasing_enabled(self, enabled=True):
        """Cho phép tắt RandomErasing ở các epoch tinh chỉnh cuối để học biên phân loại sắc nét."""
        self.erasing_enabled = bool(enabled)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        target = self.y[idx]
        if getattr(target, "ndim", 0) > 0:
            label = torch.tensor(target, dtype=torch.float32)
            primary_cls = int(np.argmax(target))
        else:
            label = torch.tensor(target, dtype=torch.long)
            primary_cls = int(target)

        if self.is_train and self.spatial_transform is not None:
            img_tensor = self.spatial_transform(img_tensor)
            # Không áp dụng RandomErasing lên mẫu thuộc 3 lớp hiếm (Disgust, Fear, Contempt)
            if (
                self.erasing_enabled
                and self.erasing_transform is not None
                and primary_cls not in self.rare_classes
            ):
                img_tensor = self.erasing_transform(img_tensor)

        if self.sample_weights is not None:
            weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
            return img_tensor, label, weight

        return img_tensor, label


def mixup_soft_batch(
    images,
    soft_labels,
    sample_weights=None,
    alpha=0.2,
    p=0.4,
    rare_classes=(5, 6, 7),
):
    """
    Kỹ thuật Selective Mixup Bảo vệ Lớp Hiếm (Rare-Class Preserving Mixup):
    - Chỉ trộn với xác suất `p` (mặc định 40% số batch).
    - Đặc biệt: Đối với các mẫu trong batch có nhãn chính thuộc nhóm lớp hiếm `rare_classes`
      (Disgust=5, Fear=6, Contempt=7), giữ nguyên hệ số trộn `lam = 1.0` (tức giữ nguyên 100% ảnh gốc
      và nhãn gốc, KHÔNG trộn với lớp đa số Neutral/Happiness).
    - Điều này ngăn chặn triệt để hiện tượng vi biểu cảm của Disgust, Contempt, Fear bị pha loãng
      với các khuôn mặt bình thản hoặc cười trong quá trình học.
    """
    if alpha <= 0.0 or p <= 0.0 or images.size(0) <= 1:
        if sample_weights is not None:
            return images, soft_labels, sample_weights
        return images, soft_labels

    if float(np.random.rand()) > p:
        if sample_weights is not None:
            return images, soft_labels, sample_weights
        return images, soft_labels

    lam = float(np.random.beta(alpha, alpha))
    # Đảm bảo ảnh chính luôn chiếm tỷ trọng lớn hơn (>= 0.5) để giữ cấu trúc khuôn mặt
    lam = max(lam, 1.0 - lam)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    hard_labels = soft_labels.argmax(dim=1)
    is_rare = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
    if rare_classes is not None:
        for rc in rare_classes:
            is_rare |= (hard_labels == int(rc))

    # Nếu mẫu hiện tại thuộc lớp hiếm (Disgust, Fear, Contempt) -> đặt lam = 1.0 (không pha trộn)
    lam_vec = torch.where(
        is_rare,
        torch.ones(batch_size, dtype=images.dtype, device=images.device),
        torch.full((batch_size,), lam, dtype=images.dtype, device=images.device),
    )

    lam_img = lam_vec.view(-1, 1, 1, 1)
    lam_lbl = lam_vec.view(-1, 1)

    mixed_images = lam_img * images + (1.0 - lam_img) * images[index]
    mixed_labels = lam_lbl * soft_labels + (1.0 - lam_lbl) * soft_labels[index]

    if sample_weights is not None:
        mixed_weights = lam_vec * sample_weights + (1.0 - lam_vec) * sample_weights[index]
        return mixed_images, mixed_labels, mixed_weights

    return mixed_images, mixed_labels