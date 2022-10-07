#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import torchvision
from packaging import version
from omnixai.preprocessing.base import TransformBase


class RandomBlur(TransformBase):
    """
    Blurs image with randomly chosen Gaussian blur.
    """

    def __init__(self, kernel_size, sigma=(0.8, 1.2)):
        super().__init__()
        self.transformer = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)

    def fit(self, x):
        return self

    def transform(self, x):
        return self.transformer(x)

    def invert(self, x):
        raise RuntimeError("`RandomBlur` doesn't support the `invert` function.")


class RandomCrop(TransformBase):
    """
    Randomly crops a batch of images.
    """
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def fit(self, x):
        return self

    def transform(self, x):
        size = (x.shape[-2] - self.shift, x.shape[-1] - self.shift)
        return torchvision.transforms.RandomCrop(size)(x)

    def invert(self, x):
        raise RuntimeError("`RandomCrop` doesn't support the `invert` function.")


class RandomResize(TransformBase):
    """
    Randomly re-sizes a batch of images.
    """

    def __init__(self, scale):
        super().__init__()
        assert isinstance(scale, (list, tuple)), \
            "`scale` should be a list or a tuple with size 2, e.g., (min_scale, max_scale)."
        self.scale = scale

    def fit(self, x):
        return self

    def transform(self, x):
        scale = (self.scale[1] - self.scale[0]) * torch.rand(1) + self.scale[0]
        h = (x.shape[-2] * scale).int()
        w = (x.shape[-1] * scale).int()
        return torchvision.transforms.Resize((h, w))(x)

    def invert(self, x):
        raise RuntimeError("`RandomResize` doesn't support the `invert` function.")


class RandomFlip(TransformBase):
    """
    Randomly flips a batch of images.
    """

    def __init__(self, horizontal=True, vertical=False):
        super().__init__()
        self.hflip = torchvision.transforms.RandomHorizontalFlip(0.5) \
            if horizontal else None
        self.vflip = torchvision.transforms.RandomVerticalFlip(0.5) \
            if vertical else None

    def fit(self, x):
        return self

    def transform(self, x):
        if self.hflip is not None:
            x = self.hflip(x)
        if self.vflip is not None:
            x = self.vflip(x)
        return x

    def invert(self, x):
        raise RuntimeError("`RandomFlip` doesn't support the `invert` function.")


class Padding(TransformBase):
    """
    Pads constant values on a batch of images.
    """

    def __init__(self, size, value=0):
        super().__init__()
        self.size = size
        self.value = value
        self.transformer = torchvision.transforms.Pad(
            padding=(size, size),
            fill=value,
            padding_mode="constant"
        )

    def fit(self, x):
        return self

    def transform(self, x):
        return self.transformer(x)

    def invert(self, x):
        raise RuntimeError("`Padding` doesn't support the `invert` function.")


def fft_images(width, height, inputs, scale):
    spectrum = torch.complex(inputs[0], inputs[1]) * scale[None, None, :, :]
    # Torch 1.7
    if version.parse(torch.__version__) < version.parse("1.8"):
        x = torch.cat([spectrum.real.unsqueeze(dim=-1), spectrum.imag.unsqueeze(dim=-1)], dim=-1)
        image = torch.irfft(x, signal_ndim=2, normalized=False, onesided=False)
    else:
        image = torch.fft.irfft2(spectrum)
    return image[:, :, :width, :height] / 4.0
