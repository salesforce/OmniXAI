#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
from PIL import Image as PilImage
from omnixai.preprocessing.image import Resize
from omnixai.data.image import Image
from omnixai.deployment.bentoml.omnixai import init_service


def test():
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../datasets/images/")
    img = Resize((256, 256)).transform(Image(PilImage.open(directory + "dog_cat.png").convert("RGB")))
    test_instance = img.to_numpy()[0]

    svc = init_service(
        model_tag="vision_explainer:latest",
        task_type="vision",
        service_name="vision_explainer"
    )
    for runner in svc.runners:
        runner.init_local()

    predictions = svc.apis["predict"].func(test_instance)
    print(predictions)
    local_explanations = svc.apis["explain"].func(test_instance, {})

    from omnixai.explainers.base import AutoExplainerBase
    exp = AutoExplainerBase.parse_explanations_from_json(local_explanations)
    for name, explanation in exp.items():
        explanation.ipython_plot()


if __name__ == "__main__":
    test()
