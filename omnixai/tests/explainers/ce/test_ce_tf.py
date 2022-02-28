#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import tensorflow as tf

from omnixai.data.image import Image
from omnixai.explainers.vision import CounterfactualExplainer


class TestCE(unittest.TestCase):
    def setUp(self) -> None:
        batch_size = 128
        num_classes = 10
        epochs = 5
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        if tf.keras.backend.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        self.x_train, self.y_train = Image(x_train.astype("float32"), batched=True), y_train
        self.x_test, self.y_test = Image(x_test.astype("float32"), batched=True), y_test
        self.preprocess_func = lambda x: np.expand_dims(x.to_numpy() / 255, axis=-1)

        x_train = self.preprocess_func(self.x_train)
        x_test = self.preprocess_func(self.x_test)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(num_classes))

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        self.model = model

    def test(self):
        explainer = CounterfactualExplainer(model=self.model, preprocess_function=self.preprocess_func)
        explanations = explainer.explain(self.x_test[0])
        explanations.plot()


if __name__ == "__main__":
    unittest.main()
