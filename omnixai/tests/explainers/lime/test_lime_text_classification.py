#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pprint
import sklearn.ensemble
from sklearn.datasets import fetch_20newsgroups
from omnixai.data.text import Text
from omnixai.explainers.nlp import LimeText
from omnixai.preprocessing.text import Tfidf
from omnixai.utils.misc import set_random_seed


class TestLimeText(unittest.TestCase):
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

        set_random_seed()
        train_vectors = self.transform.transform(self.x_train)
        test_vectors = self.transform.transform(self.x_test)
        self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        self.model.fit(train_vectors, self.y_train)
        self.predict_function = lambda x: self.model.predict_proba(self.transform.transform(x))

        predictions = self.model.predict(test_vectors)
        print("Test accuracy: {}".format(sklearn.metrics.f1_score(self.y_test, predictions, average="binary")))

    def test_explain(self):
        set_random_seed()
        idx = 83
        explainer = LimeText(predict_function=self.predict_function)
        explanations = explainer.explain(self.x_test[idx: idx + 4], num_features=6)

        for i, e in enumerate(explanations.get_explanations()):
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["tokens"], e["scores"])))
            if i == 0:
                self.assertEqual(e["target_label"], 0)
                self.assertEqual(e["tokens"][0], "NNTP")
                self.assertAlmostEqual(e["scores"][0], 0.1417, delta=1e-3)
                self.assertEqual(e["tokens"][1], "Host")
                self.assertAlmostEqual(e["scores"][1], 0.1275, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
