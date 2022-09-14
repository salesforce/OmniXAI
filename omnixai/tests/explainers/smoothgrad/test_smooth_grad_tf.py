#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import json
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision.specific.smoothgrad import SmoothGrad


class TestSmoothGrad(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        self.img = Resize((224, 224)).transform(
            Image(PilImage.open(os.path.join(directory, "images/dog_cat_2.png")).convert("RGB")))
        self.model = resnet50.ResNet50(weights="imagenet", classes=1000)

        def _preprocess(images):
            data = []
            for i in range(len(images)):
                im = tf.keras.preprocessing.image.img_to_array(images[i].to_pil())
                data.append(np.expand_dims(im, axis=0))
            data = np.concatenate(data, axis=0)
            resnet50.preprocess_input(data)
            return data

        self.preprocess = _preprocess
        with open(os.path.join(directory, "images/imagenet_class_index.json"), "r") as read_file:
            class_idx = json.load(read_file)
            self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def test_explain(self):
        explainer = SmoothGrad(
            model=self.model,
            preprocess_function=self.preprocess
        )
        explanations = explainer.explain(self.img)
        explanations.plot(class_names=self.idx2label)


if __name__ == "__main__":
    unittest.main()
