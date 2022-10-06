#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import torch
from PIL import Image as PilImage
from omnixai.preprocessing.image import Resize
from omnixai.data.image import Image
from omnixai.deployment.bentoml.omnixai import init_service


class TestService(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/images/")
        self.img = Resize((256, 256)).transform(Image(PilImage.open(directory + "dog_cat.png").convert("RGB")))

    def test(self):
        svc = init_service(
            model_tag="vision_explainer:latest",
            task_type="vision",
            service_name="vision_explainer"
        )
        for runner in svc.runners:
            runner.init_local()

        predictions = svc.apis["predict"].func(self.img)
        print(predictions)
        local_explanations = svc.apis["explain"].func(self.img, {})
        print(local_explanations)

        import json
        from omnixai.explanations.base import ExplanationBase
        d = json.loads(local_explanations)
        ExplanationBase.from_json(json.dumps(d["gradcam"])).ipython_plot()
        ExplanationBase.from_json(json.dumps(d["layercam"])).ipython_plot()


if __name__ == "__main__":
    unittest.main()
