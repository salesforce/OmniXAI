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
from omnixai.explainers.tabular import LimeTabular
from omnixai.tests.explainers.tasks import TabularClassification


class TestLimeTabular(unittest.TestCase):
    def test_1(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = LimeTabular(
            training_data=task.train_data,
            predict_function=predict_function,
            ignored_features=None,
            kernel_width=3
        )

        set_random_seed()
        i = 1653
        test_x = task.test_data.iloc([i, i + 1])
        explanations = explainer.explain(test_x)
        for i, e in enumerate(explanations.get_explanations()):
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            if i == 0:
                self.assertEqual(e["target_label"], 0)
                self.assertEqual(e["features"][0], "Capital Gain")
                self.assertAlmostEqual(e["scores"][0], 0.7206, delta=1e-3)
                self.assertEqual(e["features"][1], "Hours per week")
                self.assertAlmostEqual(e["scores"][1], 0.0901, delta=1e-3)
                self.assertEqual(e["features"][2], "Capital Loss")
                self.assertAlmostEqual(e["scores"][2], 0.0689, delta=1e-3)
            else:
                self.assertEqual(e["target_label"], 1)
                self.assertEqual(e["features"][0], "Capital Gain")
                self.assertAlmostEqual(e["scores"][0], -0.7068, delta=1e-3)
                self.assertEqual(e["features"][1], "Marital Status")
                self.assertAlmostEqual(e["scores"][1], 0.1275, delta=1e-3)
                self.assertEqual(e["features"][2], "Hours per week")
                self.assertAlmostEqual(e["scores"][2], -0.0851, delta=1e-3)

    def test_2(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = LimeTabular(training_data=task.train_data, predict_function=predict_function)

        set_random_seed()
        test_x = task.test_data.iloc(12)
        explanations = explainer.explain(test_x)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            self.assertEqual(e["target_label"], 1)
            self.assertEqual(e["features"][0], "petal length (cm)")
            self.assertAlmostEqual(e["scores"][0], 0.3304, delta=1e-3)
            self.assertEqual(e["features"][1], "petal width (cm)")
            self.assertAlmostEqual(e["scores"][1], 0.0330, delta=1e-3)

    def test_3(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_agaricus()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = LimeTabular(training_data=task.train_data, predict_function=predict_function)

        set_random_seed()
        i = 137
        test_x = task.test_data.iloc(i)
        explanations = explainer.explain(test_x, num_features=5)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            self.assertEqual(e["target_label"], 1)
            self.assertEqual(e["features"][0], "odor")
            self.assertAlmostEqual(e["scores"][0], 0.3363, delta=1e-3)
            self.assertEqual(e["features"][1], "gill-size")
            self.assertAlmostEqual(e["scores"][1], 0.3181, delta=1e-3)

    def test_4(self):
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = LimeTabular(
            training_data=task.train_data, predict_function=predict_function, random_state=1234)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = LimeTabular.load(directory=directory)

        set_random_seed()
        test_x = task.test_data.iloc(12)
        explanations = explainer.explain(test_x)
        for e in explanations.get_explanations():
            print(e["instance"])
            print(f"Target label: {e['target_label']}")
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            self.assertEqual(e["target_label"], 1)
            self.assertEqual(e["features"][0], "petal length (cm)")
            self.assertAlmostEqual(e["scores"][0], 0.3332, delta=1e-3)
            self.assertEqual(e["features"][1], "petal width (cm)")
            self.assertAlmostEqual(e["scores"][1], 0.0591, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
