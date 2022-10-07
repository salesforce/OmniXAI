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

from omnixai.explainers.vision import VisionExplainer
from omnixai.deployment.bentoml.omnixai import save_model, load_model


class TestBentoML(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/images/")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def _preprocess(ims):
            import torch
            return torch.stack([self.transform(im.to_pil()) for im in ims]).to(device)

        def _postprocess(logits):
            import torch
            return torch.nn.functional.softmax(logits, dim=1)

        self.preprocess = _preprocess
        self.model = models.resnet50(pretrained=True).to(device)
        self.postprocess = _postprocess

        with open(directory + "imagenet_class_index.json", "r") as read_file:
            class_idx = json.load(read_file)
            self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def test_save_and_load(self):
        explainer = VisionExplainer(
            explainers=["gradcam", "layercam"],
            mode="classification",
            model=self.model,
            preprocess=self.preprocess,
            postprocess=self.postprocess,
            params={
                "gradcam": {"target_layer": self.model.layer4[-1]},
                "layercam": {"target_layer": self.model.layer3[-3]},
            },
        )
        save_model("vision_explainer", explainer)
        print("Save explainer successfully.")
        explainer = load_model("vision_explainer:latest")
        print(explainer)
        print("Load explainer successfully.")


if __name__ == "__main__":
    unittest.main()
