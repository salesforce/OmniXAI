#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.feature_visualization.visualizer import \
    FeatureMapVisualizer


class TestFeatureMap(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        self.img = Image(PilImage.open(os.path.join(directory, "images/dog_cat.png")).convert("RGB"))
        self.model = models.resnet50(pretrained=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.preprocess = lambda ims: torch.stack([self.transform(im.to_pil()) for im in ims])

    def test_explain(self):
        explainer = FeatureMapVisualizer(
            model=self.model,
            target_layer=self.model.layer1[-1],
            preprocess_function=self.preprocess
        )
        explanations = explainer.explain(self.img)
        explanations.plotly_plot().show()


if __name__ == "__main__":
    unittest.main()
