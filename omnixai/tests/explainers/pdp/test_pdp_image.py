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
from omnixai.explainers.vision import PartialDependenceImage


class TestSensitivity(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img = Image(PilImage.open(os.path.join(directory, "images/dog_cat.png")).convert("RGB"))
        self.model = models.inception_v3(pretrained=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def batch_predict(images):
            self.model.eval()
            self.model.to(device)
            batch = torch.stack([self.transform(img.to_pil()) for img in images])
            batch = batch.to(device)
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        self.predict_function = batch_predict
        with open(os.path.join(directory, "images/imagenet_class_index.json"), "r") as read_file:
            class_idx = json.load(read_file)
            self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def test_explain(self):
        explainer = PartialDependenceImage(predict_function=self.predict_function)
        explanations = explainer.explain(self.img)
        explanations.plot(class_names=self.idx2label)


if __name__ == "__main__":
    unittest.main()
