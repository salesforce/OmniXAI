#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import sklearn
import sklearn.datasets
import sklearn.ensemble
import xgboost
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.deployment.bentoml.omnixai import save_model, load_model


def train():
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
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../datasets")
    tabular_data = Tabular(
        np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
        feature_columns=feature_names,
        categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
        target_column="label",
    )

    # Train an XGBoost model
    np.random.seed(1)
    transformer = TabularTransform().fit(tabular_data)
    x = transformer.transform(tabular_data)
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        x[:, :-1], x[:, -1], train_size=0.80
    )
    print("Training data shape: {}".format(train.shape))
    print("Test data shape:     {}".format(test.shape))

    gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    gbtree.fit(train[:2000], labels_train[:2000])
    print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))))

    preprocess = lambda z: transformer.transform(z)
    explainer = TabularExplainer(
        explainers=["lime", "shap", "mace", "pdp", "ale"],
        mode="classification",
        data=tabular_data,
        model=gbtree,
        preprocess=preprocess,
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
    train()
