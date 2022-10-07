#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import json
import unittest
import requests
from PIL import Image as PilImage

from omnixai.preprocessing.image import Resize
from omnixai.data.image import Image


class TestVisionRequest(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/images/")
        self.img = Resize((256, 256)).transform(
            Image(PilImage.open(directory + "dog_cat.png").convert("RGB")))

    def test(self):
        result = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "application/json"},
            data=json.dumps(self.img.to_numpy().tolist())
        ).text
        print(result)


if __name__ == "__main__":
    unittest.main()
