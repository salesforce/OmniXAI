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
from omnixai.explainers.tabular.agnostic.L2X.l2x import L2XTabular
from omnixai.tests.explainers.tasks import TabularClassification


class TestL2XTabular(unittest.TestCase):
    def test_1(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = L2XTabular(training_data=task.train_data, predict_function=predict_function)

        set_random_seed()
        i = 1653
        test_x = task.test_data.iloc(slice(i, i + 5))
        explanations = explainer.explain(test_x)
        for i, e in enumerate(explanations.get_explanations()):
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))

    def test_2(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = L2XTabular(training_data=task.train_data, predict_function=predict_function)

        set_random_seed()
        i = 12
        test_x = task.test_data.iloc(i)
        explanations = explainer.explain(test_x)
        for i, e in enumerate(explanations.get_explanations()):
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))

    def test_3(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = L2XTabular(training_data=task.train_data, predict_function=predict_function)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = L2XTabular.load(directory=directory)

        set_random_seed()
        i = 12
        test_x = task.test_data.iloc(i)
        explanations = explainer.explain(test_x)
        for i, e in enumerate(explanations.get_explanations()):
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))


if __name__ == "__main__":
    unittest.main()
