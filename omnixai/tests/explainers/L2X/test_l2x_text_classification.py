#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
import sklearn.ensemble
from sklearn.datasets import fetch_20newsgroups
from omnixai.data.text import Text
from omnixai.preprocessing.text import Tfidf
from omnixai.explainers.nlp.agnostic.l2x import L2XText


class TestL2XText(unittest.TestCase):
    def setUp(self) -> None:
        categories = ["alt.atheism", "soc.religion.christian"]
        newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
        newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
        self.newsgroups_test = newsgroups_test

        self.x_train = Text(newsgroups_train.data)
        self.y_train = newsgroups_train.target
        self.x_test = Text(newsgroups_test.data)
        self.y_test = newsgroups_test.target
        self.class_names = ["atheism", "christian"]
        self.transform = Tfidf().fit(self.x_train)

        np.random.seed(1)
        train_vectors = self.transform.transform(self.x_train)
        test_vectors = self.transform.transform(self.x_test)
        self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        self.model.fit(train_vectors, self.y_train)
        self.predict_function = lambda x: self.model.predict_proba(self.transform.transform(x))

        predictions = self.model.predict(test_vectors)
        print("Test accuracy: {}".format(sklearn.metrics.f1_score(self.y_test, predictions, average="binary")))

    def test_explain(self):
        idx = 83
        explainer = L2XText(training_data=self.x_train, predict_function=self.predict_function)
        explanations = explainer.explain(self.x_test[idx: idx + 9])
        explanations.plot(class_names=self.class_names, max_num_subplots=9)
        '''
        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = L2XText.load(directory=directory)
        explanations = explainer.explain(self.x_test[idx: idx + 9])
        explanations.plot(class_names=self.class_names, max_num_subplots=9)
        '''


if __name__ == "__main__":
    unittest.main()
