#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import torchvision
import numpy as np
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.preprocessing.image import Scale, Resize


class TestImage(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.dirname(os.path.abspath(__file__)) + "/../datasets"
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(directory, "tmp"), train=True, download=True)
        self.cifar = trainset.data
        trainset = torchvision.datasets.MNIST(root=os.path.join(directory, "tmp"), train=True, download=True)
        self.mnist = trainset.data.numpy()
        self.img = PilImage.open(os.path.join(directory, "images/dog.jpg"))
        self.cat_img = PilImage.open(os.path.join(directory, "images/cat.jpg"))

    def test_scale(self):
        data = Image(self.cifar, batched=True)
        data = Scale().transform(data)
        self.assertEqual(np.min(data.values), 0.0)
        self.assertEqual(np.max(data.values), 1.0)

    def test_resize_1(self):
        data = Image(self.img)
        transformer = Resize(size=(360, 240))
        x = transformer.transform(data)
        y = transformer.invert(x)
        self.assertEqual(x.shape, (1, 360, 240, 3))
        self.assertEqual(y.shape, (1, 720, 480, 3))

    def test_resize_2(self):
        data = Image(self.mnist[:2], batched=True)
        transformer = Resize(size=(112, 112))
        x = transformer.transform(data)
        y = transformer.invert(x)
        self.assertEqual(x.shape, (2, 112, 112, 1))
        self.assertEqual(y.shape, (2, 28, 28, 1))

    def test_resize_3(self):
        data = Image(self.img)
        transformer = Resize(size=120)
        x = transformer.transform(data)
        y = transformer.invert(x)
        self.assertEqual(x.shape, (1, 180, 120, 3))
        self.assertEqual(y.shape, (1, 720, 480, 3))

        data = Image(self.cat_img)
        transformer = Resize(size=240)
        x = transformer.transform(data)
        y = transformer.invert(x)
        self.assertEqual(x.shape, (1, 240, 338, 3))
        self.assertEqual(y.shape, (1, 354, 500, 3))


if __name__ == "__main__":
    unittest.main()
