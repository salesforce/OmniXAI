#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from sklearn.datasets import fetch_20newsgroups
from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp.specific.ig import IntegratedGradientText
from omnixai.explanations.base import ExplanationBase


class _ModelBase(nn.Module):
    def __init__(self, num_embeddings, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = kwargs.get("embedding_size", 50)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.embedding.weight.data.normal_(mean=0.0, std=0.01)


class TextModel(_ModelBase):
    def __init__(self, num_embeddings, num_classes, **kwargs):
        super().__init__(num_embeddings, **kwargs)
        hidden_size = kwargs.get("hidden_size", 100)
        kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes]

        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(self.embedding_size, hidden_size, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, num_classes)

    def forward(self, inputs, masks):
        """
        :param inputs: The input IDs.
        :param masks: The input masks.
        """
        embeddings = self.embedding(inputs)
        x = embeddings * masks.unsqueeze(dim=-1)
        x = x.permute(0, 2, 1)
        x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
        outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(dim=1)
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.load_imdb()
        except:
            self.load_newsgroups()

        self.transform = Word2Id().fit(self.x_train)
        self.model = TextModel(num_embeddings=self.transform.vocab_size, num_classes=len(self.class_names)).to(
            self.device
        )
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
        self.train()
        self.evaluate()

    def train(self):
        Trainer(optimizer_class=torch.optim.AdamW, learning_rate=1e-3,
                batch_size=128, num_epochs=10).train(
            model=self.model,
            loss_func=nn.CrossEntropyLoss(),
            train_x=self.transform.transform(self.x_train),
            train_y=self.y_train,
            padding=True,
            max_length=self.max_length,
            verbose=True,
        )

    def evaluate(self):
        self.model.eval()
        data = self.transform.transform(self.x_test)
        data_loader = DataLoader(
            dataset=InputData(data, [0] * len(data), self.max_length),
            batch_size=32,
            collate_fn=InputData.collate_func,
            shuffle=False,
        )
        outputs = []
        for inputs in data_loader:
            value, mask, target = inputs
            value = value.to(self.device)
            mask = mask.to(self.device)
            y = self.model(value, mask)
            outputs.append(y.detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        predictions = np.argmax(outputs, axis=1)
        print("Test accuracy: {}".format(
            sklearn.metrics.f1_score(self.y_test, predictions, average="binary")))

    def test_explain(self):
        idx = 83
        explainer = IntegratedGradientText(
            model=self.model,
            embedding_layer=self.model.embedding,
            preprocess_function=self.preprocess,
            id2token=self.transform.id_to_word,
        )
        explanations = explainer.explain(self.x_test[idx: idx + 9])
        explanations.plot(class_names=self.class_names, max_num_subplots=9)
        '''
        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = IntegratedGradientText.load(directory=directory)
        explanations = explainer.explain(self.x_test[idx: idx + 9])
        explanations.plot(class_names=self.class_names, max_num_subplots=9)
        '''
        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())


if __name__ == "__main__":
    unittest.main()
