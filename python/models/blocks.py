import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channels_last"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_norm(norm, channels):
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    elif norm == "ln":
        return LayerNorm(channels, data_format="channels_first")
    else:
        raise ValueError(f"Unsupported normalization type: {norm}")


def get_act(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation type: {act}")


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = get_norm(norm, out_channels)
        self.act = get_act(act)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = get_norm(norm, out_channels)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = get_norm(norm, out_channels)
        self.act = get_act(act)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = get_norm(norm, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.norm = get_norm(norm, out_channels)
        self.act = get_act(act)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
        )
        self.norm = get_norm(norm, in_channels)
        self.pointwise1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = get_act(act)
        self.pointwise2 = nn.Linear(4 * in_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.dw_conv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = x.permute(0, 3, 1, 2)
        x = x + shortcut
        return x
