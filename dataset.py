import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FER2013Dataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = X
        self.y = y
        self.is_train = is_train
        
        # Cấu hình Augmentation cho tập Train (đã giảm mức độ biến đổi để giữ cấu trúc mặt 48x48)
        if self.is_train:
            self.transform = transforms.Compose([
                # Co giãn ±10%, Dịch ±10%, Xoay ±10°
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # Lật ngang 50%
                transforms.RandomHorizontalFlip(p=0.5),
                # Xóa ngẫu nhiên với xác suất 25%, diện tích từ 2% đến 10%
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
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
            
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img_tensor, label