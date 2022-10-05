#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import json
import unittest
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam import GradCAM
from omnixai.explanations.base import ExplanationBase


class TestGradCAM(unittest.TestCase):
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

        with open(os.path.join(directory, "images/imagenet_class_index.json"), "r") as read_file:
            class_idx = json.load(read_file)
            self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def test_classification(self):
        self.model.eval()
        input_img = self.transform(self.img.to_pil()).unsqueeze(dim=0)
        probs_top_5 = torch.nn.functional.softmax(self.model(input_img), dim=1).topk(5)
        r = tuple(
            (p, c, self.idx2label[c])
            for p, c in zip(probs_top_5[0][0].detach().numpy(), probs_top_5[1][0].detach().numpy())
        )
        print(r)

    def test_explain(self):
        explainer = GradCAM(model=self.model, target_layer=self.model.layer4[-1], preprocess_function=self.preprocess)
        explanations = explainer.explain(self.img)
        explanations.plot(class_names=self.idx2label)

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()


if __name__ == "__main__":
    unittest.main()
