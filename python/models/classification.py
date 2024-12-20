import torch
import torch.nn as nn
from .blocks import BaseBlock, ResNetBlock, MobileNetBlock, ConvNeXtBlock


class BaseModle(torch.nn.Module):
    def __init__(
        self,
        block_type: str,
        channels: list,
        num_blocks: int,
        norm_type: str,
        act_type: str,
        num_classes: int,
    ):
        super().__init__()

        if block_type == "base":
            self.block = BaseBlock
        elif block_type == "resnet":
            self.block = ResNetBlock
        elif block_type == "mobilenet":
            self.block = MobileNetBlock
        elif block_type == "convnext":
            self.block = ConvNeXtBlock
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.body = self._make_body(channels, num_blocks, norm_type, act_type)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def _make_body(self, channels, num_blocks, norm_type, act_type):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                self.block(channels[i], channels[i + 1], norm_type, act_type)
            )
            for _ in range(num_blocks - 1):
                layers.append(
                    self.block(
                        channels[i + 1], channels[i + 1], norm_type, act_type
                    )
                )
            if i < len(channels) - 2:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x
