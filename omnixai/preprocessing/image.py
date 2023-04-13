#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The transformations for image data.
"""
import warnings
import numpy as np
from typing import Sequence, Union
from PIL import Image as PilImage

from .base import TransformBase
from ..data.image import Image


class Scale(TransformBase):
    """
    Rescales image pixel values to values * ratio.
    """

    def __init__(self, ratio: float = 1.0 / 255):
        super().__init__()
        assert ratio != 0, "The ratio cannot be zero."
        self.ratio = ratio

    def fit(self, x: Image) -> TransformBase:
        return self

    def transform(self, x: Image) -> Image:
        return Image(x.values * self.ratio, batched=True, channel_last=True)

    def invert(self, x: Image) -> Image:
        return Image(x.values / self.ratio, batched=True, channel_last=True)


class Round2Int(TransformBase):
    """
    Rounds float values to integer values.
    """

    def __init__(self):
        super().__init__()

    def fit(self, x: Image) -> TransformBase:
        return self

    def transform(self, x: Image) -> Image:
        return Image(np.round(x.values).astype(np.uint8), batched=True, channel_last=True)

    def invert(self, x: Image) -> Image:
        return x


class Normalize(TransformBase):
    """
    Normalizes an image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        """
        :param mean: A mean for all the channels or a sequence of means for each channel.
        :param std: A std for all the channels or a sequence of stds for each channel.
        """
        super().__init__()
        self.mean = np.array(mean)
        self.std = np.array(std)

    def fit(self, x: Image) -> TransformBase:
        return self

    def transform(self, x: Image) -> Image:
        return Image((x.values - self.mean) / self.std, batched=True, channel_last=True)

    def invert(self, x: Image) -> Image:
        return Image(x.values * self.std + self.mean, batched=True, channel_last=True)


class Resize(TransformBase):
    """
    Resizes the input image to a given size.
    """

    def __init__(self, size: Union[Sequence, int], resample=PilImage.BILINEAR):
        """
        :param size: The desired output size. If `size` is a sequence (h, w),
            the output size will be (h, w). If `size` is an int, the smaller edge
            will match this number.
        :param resample: The desired resampling strategy.
        """
        super().__init__()
        self.size = size
        self.resample = resample
        self.original_size = None

    def fit(self, x) -> TransformBase:
        return self

    def transform(self, x: Image) -> Image:
        if np.max(x.values) <= 1:
            warnings.warn("`Resize` requires an image with scale [0, 255] instead of [0, 1].")
        self.original_size = x.shape[1:3]

        if not isinstance(self.size, int):
            size = self.size
        else:
            h, w = x.shape[1:3]
            if (h <= w and h == self.size) or (w <= h and w == self.size):
                size = (h, w)
            elif h < w:
                size = (self.size, int(self.size / h * w))
            else:
                size = (int(self.size / w * h), self.size)

        data = np.zeros((x.shape[0], size[0], size[1], x.shape[-1]), dtype=np.uint8)
        values = Round2Int().transform(x).to_numpy(copy=False)
        for i in range(values.shape[0]):
            img = np.array(PilImage.fromarray(values[i]).resize(size[::-1], resample=self.resample))
            data[i] = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
        return Image(data=data, batched=True, channel_last=True)

    def invert(self, x: Image) -> Image:
        assert self.original_size is not None, "`transform` should be called before `invert`."
        return Resize(self.original_size, self.resample).transform(x)
