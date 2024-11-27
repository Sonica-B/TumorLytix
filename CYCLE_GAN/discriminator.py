import torch
import torch.nn as nn

class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        print(f"Shape after Block3D: {x.shape}")
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super().__init__()
        # First layer (no instance norm)
        self.start = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # Intermediate layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block3D(in_channels, feature, stride=2 if feature == features[-1] else 1))
            in_channels = feature

        # Final layer (output channel = 1)
        layers.append(nn.Conv3d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.start(x)
        print(f"Shape after start: {x.shape}")
        x = self.model(x)
        print(f"Shape after final layer: {x.shape}")
        return torch.sigmoid(x)

# Test with 3D input
def test():
    x = torch.randn((5, 1, 64, 256, 256))  # Example input: [Batch, Channel, Depth, Height, Width]
    model = Discriminator3D(in_channels=1)
    pred = model(x)
    print(pred.shape)  # Output should match PatchGAN principles

if __name__ == "__main__":
    test()
