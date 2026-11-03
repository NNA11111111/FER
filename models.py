import math
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ECABlock(nn.Module):
    """
    Khối Efficient Channel Attention (Wang et al., CVPR 2020):
    Sử dụng tích chập 1D cục bộ dọc theo chiều kênh mà KHÔNG giảm chiều (no dimensionality reduction),
    giúp bảo toàn mối tương quan kênh tinh tế cho các vi biểu cảm với chi phí tham số gần như bằng 0.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(3, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class SpatialMicroAttention(nn.Module):
    """
    Khối Chú ý Không gian Vi biểu cảm có Cổng Tắt (Residual Spatial Micro-Attention):
    - Làm nổi bật toạ độ vùng mắt, chân mày nhíu (Fear/Anger), sống mũi nhăn (Disgust) và khoé miệng
      nhếch lệch (Contempt) trên ảnh 48x48.
    - Sử dụng cổng tắt `x * (0.5 + 0.5 * scale)` thay vì nhân trực tiếp `x * scale` để đảm bảo
      tín hiệu nền không bao giờ bị suy giảm dưới 50% khi đi qua nhiều khối tích chập liên tiếp.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(spatial_map)
        attended = x * (0.5 + 0.5 * scale)
        return (attended, scale) if return_attention else attended


class SEBlock(nn.Module):
    """
    Khối Squeeze-and-Excitation (Hu et al., CVPR 2018):
    Học trọng số chọn lọc kênh toàn cục.
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
    Khối Tích chập Tự xây dựng (Custom Residual Attention Block) đã khắc phục lỗi Suy giảm Tín hiệu:
    - Ở phiên bản cũ, `out` bị nhân nối tiếp qua 3 cổng Sigmoid (`se` -> `eca` -> `spatial_attn`),
      làm biên độ đặc trưng vi biểu cảm bị bóp nghẹt về gần 0.
    - Ở phiên bản mới: Kết hợp SONG SONG `ECA` (85%) và `SE` (15%) trong 1 bước duy nhất (vẫn giữ nguyên
      tương thích `state_dict` 100%), sau đó đi qua `SpatialMicroAttention` có cổng tắt `0.5 + 0.5 * scale`.
    """
    def __init__(self, in_channels, out_channels, pool=True, use_eca=True, use_spatial_attn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=16)
        self.eca = ECABlock(out_channels) if use_eca else None
        self.spatial_attn = SpatialMicroAttention(kernel_size=3) if use_spatial_attn else None

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x, return_attention=False):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Kết hợp song song thay vì nhân chồng nối tiếp 2 tầng Channel Attention
        if self.eca is not None:
            out = 0.85 * self.eca(out) + 0.15 * self.se(out)
        else:
            out = self.se(out)

        attention = None
        if self.spatial_attn is not None:
            out, attention = self.spatial_attn(out, return_attention=True)

        out = self.act2(out + residual)
        out = self.pool(out)
        return (out, attention) if return_attention else out


class CustomCNN(nn.Module):
    """
    Kiến trúc CustomCNN tự thiết kế chuyên biệt cho ảnh mặt 48x48 (FER2013 / FER+):
    - Block 1: (1, 48, 48)   -> (64, 24, 24)
    - Block 2: (64, 24, 24)  -> (128, 12, 12)
    - Block 3: (128, 12, 12) -> (256, 6, 6)
    - Block 4: (256, 6, 6)   -> (512, 6, 6) [Không MaxPool -> bảo toàn bản đồ đặc trưng 6x6]
    - Multi-Scale Pooling Head: Ghép nối Global Average Pooling (512) + Global Max Pooling (512) -> 1024 chiều.
    - Giảm Dropout ở tầng phân loại từ 0.4 xuống 0.3 để tránh làm mất đặc trưng của các lớp hiếm.
    - Hỗ trợ `freeze_backbone(freeze=True, unfreeze_block4=True)` để ở Giai đoạn 2 có thể tinh chỉnh
      đồng thời cả `block4` (bản đồ đặc trưng 6x6) lẫn `classifier` trên ảnh gốc sạch.
    """
    def __init__(self, num_classes=8, use_eca=True, use_spatial_attn=True, dropout_p=0.3):
        super(CustomCNN, self).__init__()

        self.block1 = ResSEBlock(1, 64, pool=True, use_eca=use_eca, use_spatial_attn=use_spatial_attn)
        self.block2 = ResSEBlock(64, 128, pool=True, use_eca=use_eca, use_spatial_attn=use_spatial_attn)
        self.block3 = ResSEBlock(128, 256, pool=True, use_eca=use_eca, use_spatial_attn=use_spatial_attn)
        self.block4 = ResSEBlock(256, 512, pool=False, use_eca=use_eca, use_spatial_attn=use_spatial_attn)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

        self.backbone_frozen = False
        self.unfreeze_block4 = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def freeze_backbone(self, freeze=True, unfreeze_block4=True):
        """
        Đóng băng các khối tích chập đầu (block1..block3) và tuỳ chọn mở khoá block4 + classifier
        ở Giai đoạn 2 để bản đồ đặc trưng 6x6 thích nghi tinh tế hơn với các lớp hiếm (Disgust, Fear, Contempt).
        """
        self.backbone_frozen = bool(freeze)
        self.unfreeze_block4 = bool(unfreeze_block4)

        frozen_blocks = (
            (self.block1, self.block2, self.block3)
            if (self.backbone_frozen and self.unfreeze_block4)
            else (self.block1, self.block2, self.block3, self.block4)
        )

        for block in (self.block1, self.block2, self.block3, self.block4):
            is_block_frozen = self.backbone_frozen and (block in frozen_blocks)
            for param in block.parameters():
                param.requires_grad = not is_block_frozen
            if is_block_frozen:
                block.eval()

        for param in self.classifier.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "backbone_frozen", False):
            self.block1.eval()
            self.block2.eval()
            self.block3.eval()
            if not getattr(self, "unfreeze_block4", False):
                self.block4.eval()
        return self

    def forward(self, x, return_attention=False):
        attention_maps = []
        for block in (self.block1, self.block2, self.block3, self.block4):
            if return_attention:
                x, attention = block(x, return_attention=True)
                if attention is not None:
                    attention_maps.append(attention)
            else:
                x = block(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_cat = torch.cat([x_avg, x_max], dim=1)

        logits = self.classifier(x_cat)
        return (logits, attention_maps) if return_attention else logits


class FacePretrainedResNet18(nn.Module):
    """112x112 FER classifier initialized from a local face-pretrained ResNet18 checkpoint.

    The checkpoint must contain a ResNet18-compatible ``state_dict``. RGB input weights
    are averaged into a single grayscale channel. A checkpoint is required so this class
    cannot silently fall back to random initialization.
    """
    def __init__(self, checkpoint_path=None, num_classes=8, load_pretrained=True):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone
        self.backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        if load_pretrained:
            if not checkpoint_path or not Path(checkpoint_path).is_file():
                raise FileNotFoundError("Provide a local face-pretrained ResNet18 checkpoint.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
            state = {k.removeprefix("module."): v for k, v in state.items() if isinstance(v, torch.Tensor)}
            conv = state.get("conv1.weight")
            if conv is not None and conv.ndim == 4 and conv.shape[1] == 3:
                state["conv1.weight"] = conv.mean(dim=1, keepdim=True)
            target = self.backbone.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in target and target[k].shape == v.shape and not k.startswith("fc.")}
            if len(compatible) < 50:
                raise ValueError("Checkpoint does not look like a compatible ResNet18 face backbone.")
            self.backbone.load_state_dict(compatible, strict=False)

    def forward(self, x):
        return self.backbone(x)
