#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import tensorflow as tf
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.feature_visualization.tf.preprocess import \
    RandomBlur, RandomCrop, RandomResize, RandomFlip, Padding
from omnixai.preprocessing.pipeline import Pipeline


class TestPreprocess(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        img = Image(PilImage.open(os.path.join(directory, "images/dog_cat.png")).convert("RGB"))
        self.img = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img.to_pil()), axis=0)

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        plt.imshow(x.numpy()[0] / 255)
        plt.show()

    def test_blur(self):
        transform = RandomBlur(kernel_size=9)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 450, 450, 3))

    def test_crop(self):
        transform = RandomCrop(shift=100)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 350, 350, 3))

    def test_resize(self):
        transform = RandomResize(scale=(0.5, 0.5))
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 225, 225, 3))

    def test_flip(self):
        transform = RandomFlip(horizontal=True, vertical=True)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 450, 450, 3))

    def test_padding(self):
        transform = Padding(size=10)
        y = transform.transform(self.img)
        self.assertEqual(y.shape, (1, 470, 470, 3))

    def test_pipeline(self):
        unit = max(int(self.img.shape[1] / 16), 1)
        pipeline = Pipeline() \
            .step(Padding(size=unit * 4)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomBlur(kernel_size=9, sigma=(1.0, 1.1))) \
            .step(RandomCrop(unit)) \
            .step(RandomCrop(unit)) \
            .step(RandomFlip())
        y = pipeline.transform(self.img)
        self.assertEqual(y.shape, (1, 170, 170, 3))


if __name__ == "__main__":
    unittest.main()
