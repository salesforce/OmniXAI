#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
from torchvision import transforms
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.feature_visualization.pytorch.preprocess import \
    RandomBlur, RandomCrop, RandomResize, RandomFlip, Padding


class TestPreprocess(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        img = Image(PilImage.open(os.path.join(directory, "images/dog_cat.png")).convert("RGB"))
        transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.img = transform(img.to_pil()).unsqueeze(dim=0)

    @staticmethod
    def _tensor_to_numpy(x):
        x = x.detach().cpu().numpy()
        x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
        return x

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        plt.imshow(TestPreprocess._tensor_to_numpy(x))
        plt.show()

    def test_blur(self):
        transform = RandomBlur(kernel_size=9)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 3, 450, 450))

    def test_crop(self):
        transform = RandomCrop(shift=100)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 3, 350, 350))

    def test_resize(self):
        transform = RandomResize(scale=(0.5, 0.5))
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 3, 225, 225))

    def test_flip(self):
        transform = RandomFlip(horizontal=True, vertical=True)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 3, 450, 450))

    def test_padding(self):
        transform = Padding(size=10)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 3, 470, 470))


if __name__ == "__main__":
    unittest.main()
