#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision.specific.feature_visualization.visualizer import \
    FeatureMapVisualizer


class TestFeatureMap(unittest.TestCase):
    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        self.img = Resize((224, 224)).transform(
            Image(PilImage.open(os.path.join(directory, "images/dog_cat_2.png")).convert("RGB")))
        self.model = vgg16.VGG16()

        def _preprocess(images):
            data = []
            for i in range(len(images)):
                im = tf.keras.preprocessing.image.img_to_array(images[i].to_pil())
                data.append(np.expand_dims(im, axis=0))
            data = np.concatenate(data, axis=0)
            vgg16.preprocess_input(data)
            return data

        self.preprocess = _preprocess

    def test_explain(self):
        explainer = FeatureMapVisualizer(
            model=self.model,
            target_layer=self.model.layers[10],
            preprocess_function=self.preprocess
        )
        explanations = explainer.explain(self.img)
        explanations.plotly_plot().show()


if __name__ == "__main__":
    unittest.main()
