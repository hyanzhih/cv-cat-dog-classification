"""自建卷积神经网络（不使用任何预训练权重）。

设计要点（针对 100 张样本的小数据场景）：
  1. 4 个「双 3x3 卷积 + BN + ReLU + 最大池化」块，逐层通道加倍、分辨率减半；
  2. 每个卷积后接 BatchNorm，缓解小批量训练的不稳定、加快收敛；
  3. 全局平均池化替代大尺寸全连接层，参数量大幅下降，抑制过拟合；
  4. Dropout(0.5) 后接单个线性分类头；
  5. 提供特征图接口，便于可解释性可视化。
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + BN + ReLU，再接 2x2 最大池化。"""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn),
            bn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn),
            bn(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CatDogCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, width: int = 32,
                 use_bn: bool = True):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.block1 = ConvBlock(3, c1, use_bn)       # 128 -> 64
        self.block2 = ConvBlock(c1, c2, use_bn)      # 64  -> 32
        self.block3 = ConvBlock(c2, c3, use_bn)      # 32  -> 16
        self.block4 = ConvBlock(c3, c4, use_bn)      # 16  -> 8
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """返回最后一个卷积块的特征图 (B, C, H, W)，用于可视化。"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.gap(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
