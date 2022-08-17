#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pprint
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import ShapTabular
from omnixai.tests.explainers.tasks import TabularClassification


class TestShapTabular(unittest.TestCase):
    def test_1(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))

        set_random_seed()
        explainer = ShapTabular(
            training_data=task.train_data,
            predict_function=predict_function,
            ignored_features=None,
            nsamples=150
        )

        i = 1653
        test_x = task.test_data.iloc(i)
        print(predict_function(test_x))
        explanations = explainer.explain(test_x, nsamples=100)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            self.assertEqual(e["target_label"], 0)
            self.assertEqual(e["features"][0], "Capital Gain")
            self.assertEqual(e["features"][1], "Marital Status")
            self.assertEqual(e["features"][2], "Race")

    def test_2(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))

        set_random_seed()
        explainer = ShapTabular(training_data=task.train_data, predict_function=predict_function, nsamples=100)

        i = 25
        test_x = task.test_data.iloc([i, i + 1])
        explanations = explainer.explain(test_x, nsamples=100)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))

    def test_3(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_agaricus()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))

        set_random_seed()
        explainer = ShapTabular(training_data=task.train_data, predict_function=predict_function, nsamples=100)

        i = 137
        test_x = task.test_data.iloc(i)
        print(predict_function(test_x))
        explanations = explainer.explain(test_x, nsamples=100)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))

    def test_4(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = ShapTabular(training_data=task.train_data, predict_function=predict_function, nsamples=100)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = ShapTabular.load(directory=directory)

        i = 25
        test_x = task.test_data.iloc([i, i + 1])
        explanations = explainer.explain(test_x, nsamples=100)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))


if __name__ == "__main__":
    unittest.main()
