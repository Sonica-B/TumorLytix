import torch
import torch.nn as nn
from typing import Type, List, Optional, Union
from torch import Tensor


class Convblock(nn.Module):
     def __init__(
          self,
          in_channels: int,
          out_channels: int,
          down: bool = True,
          use_act: bool = True,
          **kwargs
     ) -> None:
          super().__init__()
          self.conv = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
               if down
               else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
               nn.InstanceNorm2d(out_channels),
               nn.ReLU(inplace=True) if use_act else nn.Identity(),
          )

     def forward(self, x: Tensor) -> Tensor:
          return self.conv(x)


class Residual(nn.Module):
     def __init__(self, channels: int) -> None:
          super().__init__()
          self.conv1 = Convblock(channels, channels, kernel_size=3, padding=1)
          self.conv2 = Convblock(channels, channels, use_act=False, kernel_size=3, padding=1)

     def forward(self, x: Tensor) -> Tensor:
          return x + self.conv2(self.conv1(x))


class Generator(nn.Module):
     def __init__(
          self,
          channels: int,
          num_features: int = 64,
          num_residual_blocks: int = 9
     ) -> None:
          super().__init__()

          self.initial = nn.Sequential(
               nn.Conv2d(channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
               nn.ReLU(inplace=True),
          )

          self.down_blocks = nn.Sequential(
               Convblock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
               Convblock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
          )

          self.residual_block = nn.Sequential(
               *[Residual(num_features * 4) for _ in range(num_residual_blocks)]
          )

          self.upsample_blocks = nn.Sequential(
               Convblock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
               Convblock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
          )

          self.final_layer = nn.Conv2d(num_features, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

          self._initialize_weights()

     def forward(self, x: Tensor) -> Tensor:
          x = self.initial(x)
          x = self.down_blocks(x)
          x = self.residual_block(x)
          x = self.upsample_blocks(x)
          return torch.tanh(self.final_layer(x))

     def _initialize_weights(self) -> None:
          for m in self.modules():
               if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
               if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def test() -> None:
     img_channels = 3
     img_size = 256
     x = torch.randn((1, img_channels, img_size, img_size))

     model = Generator(channels=img_channels)
     out = model(x)
     print("Input Shape:", x.shape)
     print("Output Shape:", out.shape)
