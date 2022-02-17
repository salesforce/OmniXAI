#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import torchvision
from PIL import Image as PilImage
from omnixai.data.image import Image


class TestImage(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.dirname(os.path.abspath(__file__))
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(directory, "../datasets/tmp"), train=True, download=True
        )
        self.cifar = trainset.data
        trainset = torchvision.datasets.MNIST(
            root=os.path.join(directory, "../datasets/tmp"), train=True, download=True
        )
        self.mnist = trainset.data.numpy()
        self.img = PilImage.open(os.path.join(directory, "../datasets/images/dog.jpg"))

    def test_1(self):
        data = Image(self.cifar, batched=True)
        assert data.shape == (50000, 32, 32, 3)
        assert data.image_shape == (32, 32, 3)
        assert data.to_numpy().shape == (50000, 32, 32, 3)
        assert data.to_numpy(hwc=False).shape == (50000, 3, 32, 32)

    def test_2(self):
        data = Image(self.mnist, batched=True)
        assert data.shape == (60000, 28, 28, 1)
        assert data.image_shape == (28, 28, 1)
        assert data.to_numpy().shape == (60000, 28, 28)
        assert data.to_numpy(keepdim=True).shape == (60000, 28, 28, 1)

    def test_3(self):
        data = Image(self.img)
        assert data.shape == (1, 720, 480, 3)
        assert data.image_shape == (720, 480, 3)
        assert data.to_numpy().shape == (1, 720, 480, 3)


if __name__ == "__main__":
    unittest.main()
