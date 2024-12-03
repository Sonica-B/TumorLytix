import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for idx in range(1, len(features)):
            layers.append(
                nn.Conv2d(
                    features[idx - 1], features[idx], kernel_size=4, stride=2 if idx != len(features) - 1 else 1, padding=1, padding_mode="reflect"
                )
            )
            layers.append(nn.InstanceNorm2d(features[idx]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
