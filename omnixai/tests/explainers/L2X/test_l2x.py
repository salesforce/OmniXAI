#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from omnixai.explainers.tabular.agnostic.L2X.utils import L2XModel, Trainer


class PredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Softplus(), nn.Linear(128, 64), nn.Softplus(), nn.Linear(64, output_dim)
        )

    def forward(self, inputs, weights):
        inputs = inputs * weights
        return self.layer(inputs).squeeze()


class TestL2X(unittest.TestCase):
    @staticmethod
    def diabetes_data(file_path="diabetes.csv"):
        data = pd.read_csv(file_path)
        data = data.replace(
            to_replace=["Yes", "No", "Positive", "Negative", "Male", "Female"], value=[1, 0, 1, 0, 1, 0]
        )
        features = [
            "Age",
            "Gender",
            "Polyuria",
            "Polydipsia",
            "sudden weight loss",
            "weakness",
            "Polyphagia",
            "Genital thrush",
            "visual blurring",
            "Itching",
            "Irritability",
            "delayed healing",
            "partial paresis",
            "muscle stiffness",
            "Alopecia",
            "Obesity",
        ]

        y = data["class"]
        data = data.drop(["class"], axis=1)
        x_train_un, x_test_un, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=2, stratify=y)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train_un)
        x_test = sc.transform(x_test_un)

        x_train = x_train.astype(np.float32)
        y_train = y_train.to_numpy()
        x_test = x_test.astype(np.float32)
        y_test = y_test.to_numpy()

        return x_train, y_train, x_test, y_test, features, x_train_un, x_test_un

    def setUp(self) -> None:
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/../../datasets/diabetes.csv"
        self.train_x, self.train_y, self.test_x, self.test_y, self.feature_names, self.x_train_un, self.x_test_un = self.diabetes_data(
            file_path
        )
        print("Training data shape: {}".format(self.train_x.shape))
        print("Test data shape:     {}".format(self.test_x.shape))

        self.selection_model = nn.Sequential(
            nn.Linear(self.train_x.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.train_x.shape[1]),
        )
        self.prediction_model = PredictionModel(input_dim=self.train_x.shape[1], output_dim=1)

    def test(self):
        model = L2XModel(selection_model=self.selection_model, prediction_model=self.prediction_model, tau=0.1, k=5)
        loss_func = nn.BCEWithLogitsLoss()
        trainer = Trainer(optimizer_class=torch.optim.Adam, learning_rate=1e-3, batch_size=64, num_epochs=100)
        trainer.train(
            model=model,
            loss_func=loss_func,
            train_x=self.train_x.astype("float32"),
            train_y=self.train_y.astype("float32"),
            valid_x=self.test_x.astype("float32"),
            valid_y=self.test_y.astype("float32"),
        )

        test_instances = self.test_x[:1].astype("float32")
        weights, selections = model.explain(torch.tensor(test_instances))
        weights = weights.detach().cpu().numpy()
        self.assertEqual(len(self.feature_names), weights.shape[1])
        feature_scores = list(zip(self.feature_names, test_instances[0], weights[0]))
        feature_scores = sorted(feature_scores, key=lambda s: s[-1], reverse=True)
        for f in enumerate(feature_scores):
            print(f)


if __name__ == "__main__":
    unittest.main()
