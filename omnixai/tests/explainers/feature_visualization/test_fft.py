#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import torch
import tensorflow as tf
from omnixai.explainers.vision.specific.feature_visualization.utils import \
    fft_freq, fft_scale, fft_inputs
from omnixai.explainers.vision.specific.feature_visualization.tf.preprocess import \
    fft_images as fft_images_tf
from omnixai.explainers.vision.specific.feature_visualization.pytorch.preprocess import \
    fft_images as fft_images_torch


class TestFFT(unittest.TestCase):

    def test_1(self):
        batch_size = 5
        channel = 3
        width = 10
        height = 7
        mode = "torch"

        freq = fft_freq(width, height, mode)
        scale = fft_scale(width, height, mode)
        inputs = fft_inputs(batch_size, channel, width, height, mode)
        self.assertEqual(freq.shape, (10, 7))
        self.assertEqual(scale.shape, (10, 7))
        self.assertEqual(inputs.shape, (2, 5, 3, 10, 7))

    def test_2(self):
        batch_size = 5
        channel = 3
        width = 10
        height = 7
        mode = "tf"

        freq = fft_freq(width, height, mode)
        scale = fft_scale(width, height, mode)
        inputs = fft_inputs(batch_size, channel, width, height, mode)
        self.assertEqual(freq.shape, (10, 5))
        self.assertEqual(scale.shape, (10, 5))
        self.assertEqual(inputs.shape, (2, 5, 3, 10, 5))

    def test_3(self):
        batch_size = 5
        channel = 3
        width = 10
        height = 7
        mode = "tf"

        scale = fft_scale(width, height, mode)
        scale = tf.convert_to_tensor(scale, dtype=tf.complex64)
        inputs = fft_inputs(batch_size, channel, width, height, mode)
        inputs = tf.convert_to_tensor(inputs)

        images = fft_images_tf(width, height, inputs, scale)
        self.assertEqual(images.shape, (5, 10, 7, 3))

    def test_4(self):
        batch_size = 5
        channel = 3
        width = 10
        height = 7
        mode = "torch"

        scale = fft_scale(width, height, mode)
        scale = torch.tensor(scale, dtype=torch.complex64)
        inputs = fft_inputs(batch_size, channel, width, height, mode)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        images = fft_images_torch(width, height, inputs, scale)
        self.assertEqual(images.shape, (5, 3, 10, 7))


if __name__ == "__main__":
    unittest.main()
