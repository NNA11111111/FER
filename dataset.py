import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FER2013Dataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = X
        self.y = y
        self.is_train = is_train
        
        # Cấu hình Augmentation cho tập Train (giữ cấu trúc mặt 48x48)
        if self.is_train:
            self.transform = transforms.Compose([
                # Co giãn ±10%, Dịch ±10%, Xoay ±10°
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # Lật ngang 50%
                transforms.RandomHorizontalFlip(p=0.5),
                # Dùng pixel ngẫu nhiên để tránh vùng xám cố định trên ảnh đã chuẩn hóa [-1, 1].
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value='random')
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Ảnh đã được chuẩn hóa [-1.0, 1.0] và có shape (48, 48, 1) từ data_utils
        img = self.X[idx] 
        # Chuyển (H, W, C) -> Tensor (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        if self.is_train:
            img_tensor = self.transform(img_tensor)
            
        # Nhãn mềm là vector phân phối xác suất; nhãn cứng là số nguyên lớp.
        target = self.y[idx]
        if getattr(target, "ndim", 0) > 0:
            label = torch.tensor(target, dtype=torch.float32)
        else:
            label = torch.tensor(target, dtype=torch.long)

        return img_tensor, label
