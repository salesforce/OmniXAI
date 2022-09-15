#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.datasets import fetch_20newsgroups
from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.nlp.specific.ig import IntegratedGradientText


class TextModel(tf.keras.Model):
    def __init__(self, num_embeddings, num_classes, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = kwargs.get("embedding_size", 50)
        hidden_size = kwargs.get("hidden_size", 100)
        kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])

        self.embedding = tf.keras.layers.Embedding(
            num_embeddings,
            self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            name="embedding",
        )
        self.conv_layers = [
            tf.keras.layers.Conv1D(hidden_size, k, activation="relu", padding="same") for k in kernel_sizes
        ]
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, masks, training=False):
        embeddings = self.embedding(inputs)
        x = embeddings * tf.expand_dims(masks, axis=-1)
        x = [tf.reduce_max(layer(x), axis=1) for layer in self.conv_layers]
        x = self.dropout(tf.concat(x, axis=1)) if training else tf.concat(x, axis=1)
        outputs = self.output_layer(x)
        return outputs


class TestIG(unittest.TestCase):
    def load_newsgroups(self):
        categories = ["alt.atheism", "soc.religion.christian"]
        newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
        newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
        self.x_train = Text(newsgroups_train.data)
        self.y_train = newsgroups_train.target
        self.x_test = Text(newsgroups_test.data)
        self.y_test = newsgroups_test.target
        self.class_names = ["atheism", "christian"]

    def load_imdb(self):
        train_data = pd.read_csv("/home/ywz/data/imdb/labeledTrainData.tsv", sep="\t")
        n = int(0.8 * len(train_data))
        self.x_train = Text(train_data["review"].values[:n])
        self.y_train = train_data["sentiment"].values[:n].astype(int)
        self.x_test = Text(train_data["review"].values[n:])
        self.y_test = train_data["sentiment"].values[n:].astype(int)
        self.class_names = ["negative", "postive"]

    def setUp(self) -> None:
        try:
            self.load_imdb()
        except:
            self.load_newsgroups()
        self.transform = Word2Id().fit(self.x_train)
        self.max_length = 256

        def _preprocess(X: Text):
            samples = self.transform.transform(X)
            max_len = 0
            for i in range(len(samples)):
                max_len = max(max_len, len(samples[i]))
            max_len = min(max_len, self.max_length)
            inputs = np.zeros((len(samples), max_len), dtype=int)
            masks = np.zeros((len(samples), max_len), dtype=np.float32)
            for i in range(len(samples)):
                x = samples[i][:max_len]
                inputs[i, : len(x)] = x
                masks[i, : len(x)] = 1
            return inputs, masks

        self.preprocess = _preprocess
        self.model = TextModel(num_embeddings=self.transform.vocab_size, num_classes=len(self.class_names))
        self.train()
        self.evaluate()

    def train(self, learning_rate=1e-3, batch_size=128, num_epochs=10):
        inputs, masks = self.preprocess(self.x_train)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        train_dataset = tf.data.Dataset.from_tensor_slices((inputs, masks, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        for epoch in range(num_epochs):
            for step, (ids, masks, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(ids, masks, training=True)
                    loss = loss_fn(labels, logits)
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                if step % 200 == 0:
                    print(f"Training loss at epoch {epoch}, step {step}: {float(loss)}")

    def evaluate(self):
        inputs, masks = self.preprocess(self.x_test)
        outputs = self.model(inputs, masks).numpy()
        predictions = np.argmax(outputs, axis=1)
        print("Test accuracy: {}".format(sklearn.metrics.f1_score(self.y_test, predictions, average="binary")))

    def test_explain(self):
        explainer = IntegratedGradientText(
            model=self.model,
            embedding_layer=self.model.embedding,
            preprocess_function=self.preprocess,
            id2token=self.transform.id_to_word,
        )
        x = Text(
            [
                "What a great movie! if you have no taste.",
                "it was a fantastic performance!",
                "best film ever",
                "such a great show!",
                "it was a horrible movie",
                "i've never watched something as bad",
            ]
        )
        explanations = explainer.explain(x)
        fig = explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()
