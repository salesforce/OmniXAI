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
import numpy as np
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.preprocessing.image import Resize
from omnixai.data.image import Image
from omnixai.explainers.vision import VisionExplainer
from omnixai.deployment.bentoml.omnixai import save_model, load_model


class TestBentoML(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/images/")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_1 = Resize((256, 256)).transform(Image(PilImage.open(directory + "dog_cat.png").convert("RGB")))
        img_2 = Resize((256, 256)).transform(Image(PilImage.open(directory + "dog.jpg").convert("RGB")))
        img_3 = Resize((256, 256)).transform(Image(PilImage.open(directory + "camera.jpg").convert("RGB")))
        self.img = Image(data=np.concatenate([img_1.to_numpy(), img_2.to_numpy(), img_3.to_numpy()]), batched=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.preprocess = lambda ims: torch.stack([self.transform(im.to_pil()) for im in ims]).to(device)
        self.model = models.resnet50(pretrained=True).to(device)
        self.postprocess = lambda logits: torch.nn.functional.softmax(logits, dim=1)

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
