import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, features: Optional[List[int]] = None) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.start = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers: List[nn.Module] = []
        in_channels = features[0]
        for feature in features[1:]:
            stride = 2 if feature != features[-1] else 1
            layers.append(Block(in_channels, feature, stride))
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.start(x)
        x = self.model(x)
        return torch.sigmoid(x)


def test() -> None:
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    pred = model(x)
    print(f"Output shape: {pred.shape}")  # Expect: [5, 1, 15, 15]
