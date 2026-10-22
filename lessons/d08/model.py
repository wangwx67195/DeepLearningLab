import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.Layers_shortcut = nn.Identity()
        else:
            self.Layers_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

        self.Layers_main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        pred_main = self.Layers_main(input)
        pred_shortcut = self.Layers_shortcut(input)
        assert pred_main.shape == pred_shortcut.shape, \
            "After forward, main_shape and shortcut_shape should be SAME !"

        pred = nn.ReLU()(pred_main + pred_shortcut)

        return pred

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            ResidualBlock(8, 8, stride = 1),
            ResidualBlock(8, 16, stride = 2)
        )

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(16, 3)

    def forward(self, input):
        pred = self.Layers(input)
        pred = self.AdaptiveAvgPool(pred)
        pred = torch.flatten(pred, start_dim = 1)
        pred = self.Linear(pred)

        return pred