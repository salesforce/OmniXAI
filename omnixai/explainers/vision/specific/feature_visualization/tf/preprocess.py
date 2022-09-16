#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import tensorflow as tf
from omnixai.preprocessing.base import TransformBase


class RandomBlur(TransformBase):
    """
    Blurs image with randomly chosen Gaussian blur
    """

    def __init__(self, kernel_size, sigma=(0.8, 1.2)):
        super().__init__()
        assert isinstance(sigma, (list, tuple)), \
            "`sigma` should be a list or a tuple with size 2, e.g., (min_sigma, max_sigma)."
        self.kernel_size = kernel_size
        self.sigma = sigma
        ranges = np.linspace(
            -(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size, dtype=np.float32)
        self.X, self.Y = tf.meshgrid(ranges, ranges)

    def fit(self, x):
        return self

    def transform(self, x):
        assert len(x.shape) == 4, \
            "`x` must be 4-dimensional."
        sigma = tf.random.uniform(
            shape=[],
            minval=self.sigma[0],
            maxval=self.sigma[1],
            dtype=tf.float32
        )
        kernel = tf.exp(-0.5 * (self.X ** 2 + self.Y ** 2) / sigma ** 2)
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel, (self.kernel_size, self.kernel_size, 1, 1))
        if x.shape[-1] == 3:
            kernel = tf.tile(kernel, [1, 1, 3, 1])
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")

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
        assert len(x.shape) == 4, \
            "`x` must be 4-dimensional."
        shape = tf.shape(x)
        return tf.image.random_crop(
            x, (shape[0], shape[1] - self.shift, shape[2] - self.shift, shape[-1]))

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
        assert len(x.shape) == 4, \
            "`x` must be 4-dimensional."
        scale = tf.random.uniform(
            shape=[],
            minval=self.scale[0],
            maxval=self.scale[1],
            dtype=tf.float32
        )
        return tf.image.resize(
            x, tf.cast([x.shape[1] * scale, x.shape[2] * scale], tf.int32))

    def invert(self, x):
        raise RuntimeError("`RandomResize` doesn't support the `invert` function.")


class RandomFlip(TransformBase):
    """
    Randomly flips a batch of images.
    """

    def __init__(self, horizontal=True, vertical=False):
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical

    def fit(self, x):
        return self

    def transform(self, x):
        assert len(x.shape) == 4, \
            "`x` must be 4-dimensional."
        if self.horizontal:
            x = tf.image.random_flip_left_right(x)
        if self.vertical:
            x = tf.image.random_flip_up_down(x)
        return x

    def invert(self, x):
        raise RuntimeError("`RandomFlip` doesn't support the `invert` function.")


class Padding(TransformBase):
    """
    Pads constant values on a batch of images.
    """

    def __init__(self, size, value=0):
        super().__init__()
        self.value = value
        self.paddings = [(0, 0), (size, size), (size, size), (0, 0)]

    def fit(self, x):
        return self

    def transform(self, x):
        assert len(x.shape) == 4, \
            "`x` must be 4-dimensional."
        return tf.pad(
            x,
            paddings=self.paddings,
            mode="CONSTANT",
            constant_values=tf.cast(self.value, tf.float32)
        )

    def invert(self, x):
        raise RuntimeError("`Padding` doesn't support the `invert` function.")


def fft_images(width, height, inputs, scale):
    spectrum = tf.complex(inputs[0], inputs[1]) * scale
    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))
    image = image[:, :width, :height, :]
    return image / 4.0
