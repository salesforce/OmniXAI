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
from omnixai.explainers.data import DataAnalyzer
from omnixai.explainers.tabular import TabularExplainer
from omnixai.explainers.prediction import PredictionAnalyzer
from omnixai.visualization.dashboard import Dashboard


class TestDashboard(unittest.TestCase):
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
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
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
        gbtree.fit(train, labels_train)
        print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))))

        self.tabular_data = tabular_data
        self.model = gbtree
        self.preprocess = lambda z: transformer.transform(z)
        self.test_data = transformer.invert(test)
        self.test_targets = labels_test

        i = 1653
        self.instances = transformer.invert(test[i: i + 5])

    def test(self):
        np.random.seed(1)
        explainer = DataAnalyzer(
            explainers=["correlation", "imbalance#0", "imbalance#1",
                        "imbalance#2", "imbalance#3", "mutual", "chi2"],
            mode="classification",
            data=self.tabular_data
        )
        data_explanations = explainer.explain(
            params={
                "imbalance#0": {"features": ["Sex"]},
                "imbalance#1": {"features": ["Race"]},
                "imbalance#2": {"features": ["Sex", "Race"]},
                "imbalance#3": {"features": ["Marital Status", "Age"]},
            }
        )

        explainer = PredictionAnalyzer(
            mode="classification",
            test_data=self.test_data,
            test_targets=self.test_targets,
            model=self.model,
            preprocess=self.preprocess
        )
        prediction_explanations = explainer.explain()

        explainers = TabularExplainer(
            explainers=["lime", "shap", "mace", "knn_ce", "pdp", "ale", "shap_global"],
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
        local_explanations = explainers.explain(X=self.instances)
        global_explanations = explainers.explain_global(
            params={
                "pdp": {
                    "features": [
                        "Age",
                        "Education-Num",
                        "Capital Gain",
                        "Capital Loss",
                        "Hours per week",
                        "Education",
                        "Marital Status",
                        "Occupation",
                    ]
                }
            }
        )

        dashboard = Dashboard(
            instances=self.instances,
            local_explanations=local_explanations,
            global_explanations=global_explanations,
            data_explanations=data_explanations,
            prediction_explanations=prediction_explanations,
            class_names=self.class_names
        )
        dashboard.show()


if __name__ == "__main__":
    unittest.main()
