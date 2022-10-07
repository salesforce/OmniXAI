#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import sklearn
import sklearn.datasets
import sklearn.ensemble
import xgboost
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.deployment.bentoml.omnixai import save_model, load_model


class TestBentoML(unittest.TestCase):

    def setUp(self) -> None:
        # Load the dataset
        feature_names = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country",
            "label",
        ]
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        tabular_data = Tabular(
            np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )

        # Train an XGBoost model
        np.random.seed(1)
        transformer = TabularTransform().fit(tabular_data)
        self.class_names = transformer.class_names
        x = transformer.transform(tabular_data)
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x[:, :-1], x[:, -1], train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
        gbtree.fit(train[:2000], labels_train[:2000])
        print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))))

        self.tabular_data = tabular_data
        self.model = gbtree
        self.preprocess = lambda z: transformer.transform(z)
        self.test_data = transformer.invert(test)
        self.test_targets = labels_test

        i = 1653
        self.instances = transformer.invert(test[i: i + 5])

    def test_save_and_load(self):
        np.random.seed(1)
        explainer = TabularExplainer(
            explainers=["lime", "shap", "mace", "pdp", "ale"],
            mode="classification",
            data=self.tabular_data,
            model=self.model,
            preprocess=self.preprocess,
            params={
                "lime": {"kernel_width": 3},
                "shap": {"nsamples": 100},
                "mace": {"ignored_features": ["Sex", "Race", "Relationship", "Capital Loss"]},
            },
        )
        save_model("tabular_explainer", explainer)
        print("Save explainer successfully.")
        explainer = load_model("tabular_explainer:latest")
        print(explainer)
        print("Load explainer successfully.")


if __name__ == "__main__":
    unittest.main()
