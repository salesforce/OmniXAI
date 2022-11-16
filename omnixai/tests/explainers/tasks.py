#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import dill
import sklearn
import sklearn.datasets
import sklearn.ensemble

import xgboost
import numpy as np
import pandas as pd
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.utils.misc import set_random_seed


class Task:
    def __init__(self, name, model=None, transform=None, data=None,
                 train_data=None, test_data=None, test_targets=None):
        self.name = name
        self.model = model
        self.transform = transform
        self.data = data
        self.train_data = train_data
        self.test_data = test_data
        self.test_targets = test_targets

    def save(self, directory):
        path = os.path.join(directory, self.name)
        with open(path, "wb") as f:
            data = {
                "model": self.model,
                "transform": self.transform,
                "data": self.data,
                "train_data": self.train_data,
                "test_data": self.test_data,
            }
            dill.dump(data, f)

    def load(self, directory):
        path = os.path.join(directory, self.name)
        with open(path, "rb") as f:
            data = dill.load(f)
            self.model = data["model"]
            self.transform = data["transform"]
            self.data = data["data"]
            self.train_data = data["train_data"]
            self.test_data = data["test_data"]


class TabularClassification:
    def __init__(self, base_folder=None):
        if base_folder is None:
            self.base_folder = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_folder = base_folder
        self.tasks = [self.train_adult, self.train_iris, self.train_agaricus]

    def train_adult(self, num_training_samples=None):
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
        data_dir = os.path.join(self.base_folder, "../datasets")
        data = np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str)
        tabular_data = Tabular(
            data,
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )

        set_random_seed()
        transformer = TabularTransform().fit(tabular_data)
        x = transformer.transform(tabular_data)
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x[:, :-1], x[:, -1], train_size=0.80
        )

        gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
        if num_training_samples is not None:
            gbtree.fit(train[:num_training_samples], labels_train[:num_training_samples])
        gbtree.fit(train, labels_train)
        print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))))

        return Task(
            name="adult",
            model=gbtree,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
            test_targets=labels_test
        )

    def train_iris(self):
        iris = sklearn.datasets.load_iris()
        tabular_data = Tabular(iris.data, feature_columns=iris.feature_names)

        set_random_seed()
        transformer = TabularTransform().fit(tabular_data)
        x = transformer.transform(tabular_data)
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x, iris.target, train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        model = sklearn.svm.SVC(kernel="rbf", probability=True)
        model.fit(train, labels_train)
        print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, model.predict(test))))

        return Task(
            name="iris",
            model=model,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
        )

    def train_agaricus(self):
        feature_names = (
            "label,cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,"
            "gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,"
            "stalk-surface-above-ring, stalk-surface-below-ring, "
            "stalk-color-above-ring,stalk-color-below-ring,veil-type,"
            "veil-color,ring-number,ring-type,spore-print-color,"
            "population,habitat".split(",")
        )
        data_dir = os.path.join(self.base_folder, "../datasets")
        data = np.genfromtxt(os.path.join(data_dir, "agaricus-lepiota.data"), delimiter=",", dtype="<U20")
        tabular_data = Tabular(
            data, feature_columns=feature_names, categorical_columns=feature_names[1:], target_column="label"
        )

        set_random_seed()
        transformer = TabularTransform().fit(tabular_data)
        x = transformer.transform(tabular_data)
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x[:, :-1], x[:, -1], train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        model = sklearn.svm.SVC(kernel="rbf", probability=True)
        model.fit(train, labels_train)
        print("Test accuracy: {}".format(sklearn.metrics.accuracy_score(labels_test, model.predict(test))))

        return Task(
            name="agaricus",
            model=model,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
        )

    def train_and_dump(self, directory):
        for task in self.tasks:
            task().save(directory)


class TabularRegression:
    def __init__(self):
        self.tasks = [self.train_boston, self.train_boston_continuous]

    def train_boston(self):
        from sklearn.datasets import load_boston

        set_random_seed()
        boston = load_boston()
        tabular_data = Tabular(
            boston.data,
            feature_columns=boston.feature_names,
            categorical_columns=[boston.feature_names[i] for i in [3, 8]],
        )
        transformer = TabularTransform().fit(tabular_data)
        x = transformer.transform(tabular_data)

        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x, boston.target, train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
        rf.fit(train, labels_train)
        print("Random Forest MSError", np.mean((rf.predict(test) - labels_test) ** 2))
        print("MSError when predicting the mean", np.mean((labels_train.mean() - labels_test) ** 2))

        return Task(
            name="boston",
            model=rf,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
            test_targets=labels_test
        )

    def train_boston_continuous(self):
        from sklearn.datasets import load_boston

        set_random_seed()
        boston = load_boston()
        df = pd.DataFrame(
            np.concatenate([boston.data, boston.target.reshape((-1, 1))], axis=1),
            columns=list(boston.feature_names) + ["target"],
        )
        df = df.drop(columns=[boston.feature_names[i] for i in [3, 8]])

        tabular_data = Tabular(df, target_column="target")
        transformer = TabularTransform(target_transform=Identity()).fit(tabular_data)
        x = transformer.transform(tabular_data)

        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x[:, :-1], x[:, -1], train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
        rf.fit(train, labels_train)
        print("Random Forest MSError", np.mean((rf.predict(test) - labels_test) ** 2))
        print("MSError when predicting the mean", np.mean((labels_train.mean() - labels_test) ** 2))

        return Task(
            name="boston_continuous",
            model=rf,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
            test_targets=labels_test
        )

    def train_california_housing(self):
        from sklearn.datasets import fetch_california_housing

        set_random_seed()
        housing = fetch_california_housing()
        df = pd.DataFrame(
            np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
            columns=list(housing.feature_names) + ["target"],
        )

        tabular_data = Tabular(df, target_column="target")
        transformer = TabularTransform(target_transform=Identity()).fit(tabular_data)
        x = transformer.transform(tabular_data)

        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            x[:, :-1], x[:, -1], train_size=0.80
        )
        print("Training data shape: {}".format(train.shape))
        print("Test data shape:     {}".format(test.shape))

        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
        rf.fit(train, labels_train)
        print("Random Forest MSError", np.mean((rf.predict(test) - labels_test) ** 2))
        print("MSError when predicting the mean", np.mean((labels_train.mean() - labels_test) ** 2))

        return Task(
            name="california_housing",
            model=rf,
            transform=transformer,
            data=tabular_data,
            train_data=transformer.invert(train),
            test_data=transformer.invert(test),
            test_targets=labels_test
        )

    def train_and_dump(self, directory):
        for task in self.tasks:
            task().save(directory)
