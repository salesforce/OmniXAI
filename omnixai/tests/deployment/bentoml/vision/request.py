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
from requests_toolbelt.multipart.encoder import MultipartEncoder


class TestVisionRequest(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/images/")
        self.img = Resize((256, 256)).transform(
            Image(PilImage.open(directory + "dog_cat.png").convert("RGB")))

    def test(self):
        data = json.dumps(self.img.to_numpy().tolist())

        result = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "application/json"},
            data=data
        ).text
        print(result)

        m = MultipartEncoder(
            fields={
                "data": data,
                "params": '{}',
            }
        )
        result = requests.post(
            "http://0.0.0.0:3000/explain",
            headers={"Content-Type": m.content_type},
            data=m
        ).text

        from omnixai.explainers.base import AutoExplainerBase
        exp = AutoExplainerBase.parse_explanations_from_json(result)
        for name, explanation in exp.items():
            explanation.ipython_plot()


if __name__ == "__main__":
    unittest.main()
