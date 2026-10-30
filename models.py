import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Khối Squeeze-and-Excitation (Hu et al., CVPR 2018):
    Nén thông tin không gian toàn cục bằng Global Average Pooling và học trọng số
    chọn lọc kênh (Channel Attention), giúp khuếch đại các kênh đặc trưng vùng mắt,
    chân mày, khoé miệng và triệt tiêu các kênh nhiễu nền.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class ResSEBlock(nn.Module):
    """
    Khối Tích chập kết hợp nhánh tắt Residual (He et al., CVPR 2016) và SE-Attention:
    - Dùng `bias=False` trước `BatchNorm2d` để loại bỏ tham số dư thừa.
    - Nhánh tắt `shortcut` (1x1 Conv) giúp gradient truyền thẳng, chống suy giảm đạo hàm
      và kết hợp đặc trưng biên mức thấp với ngữ nghĩa mức cao.
    - Không đặt `Dropout2d` xen giữa các khối `BatchNorm2d` để tránh hiện tượng lệch phương sai
      (Variance Shift - Li et al., CVPR 2019) giữa lúc Train và Eval.
    """
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=16)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = self.act2(out + residual)
        out = self.pool(out)
        return out


class CustomCNN(nn.Module):
    """
    Phương án C hoàn chỉnh — Kiến trúc CustomResSECNN (giữ tên class CustomCNN để tương thích toàn bộ dự án):
    - Block 1: (1, 48, 48)   -> (64, 24, 24)  [ResSEBlock + MaxPool2d]
    - Block 2: (64, 24, 24)  -> (128, 12, 12) [ResSEBlock + MaxPool2d]
    - Block 3: (128, 12, 12) -> (256, 6, 6)   [ResSEBlock + MaxPool2d]
    - Block 4: (256, 6, 6)   -> (512, 6, 6)   [ResSEBlock không MaxPool -> bảo toàn bản đồ đặc trưng 6x6]
    - Multi-Scale Pooling Head: Ghép nối Global Average Pooling (512) + Global Max Pooling (512) -> 1024 chiều
      rồi qua tầng phân loại tinh gọn (chỉ ~5.1M tham số thay vì ~7.05M như mạng cũ, chống overfit mạnh mẽ).
    """
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()

        self.block1 = ResSEBlock(1, 64, pool=True)
        self.block2 = ResSEBlock(64, 128, pool=True)
        self.block3 = ResSEBlock(128, 256, pool=True)
        # Giữ độ phân giải không gian 6x6 ở Block 4 để không làm mất chi tiết khoé miệng / chân mày
        self.block4 = ResSEBlock(256, 512, pool=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Kết hợp đặc trưng trung bình toàn cục (ngữ cảnh khuôn mặt) và cực đại toàn cục (cơ mặt nổi bật nhất)
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_cat = torch.cat([x_avg, x_max], dim=1)

        return self.classifier(x_cat)