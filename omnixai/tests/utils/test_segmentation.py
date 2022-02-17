#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
from PIL import Image as PilImage
from omnixai.data.image import Image
from omnixai.utils.segmentation import image_segmentation


class TestSegmentation(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/images/")
        self.img = Image(PilImage.open(directory + "dog.jpg").convert("RGB"))

    def test(self):
        mask = image_segmentation(image=self.img.to_numpy()[0], method="grid")
        self.assertEqual(np.min(mask), 0)
        self.assertEqual(np.max(mask), 149)


if __name__ == "__main__":
    unittest.main()
