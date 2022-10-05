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
from omnixai.explainers.tabular import PartialDependenceTabular
from omnixai.tests.explainers.tasks import TabularClassification
from omnixai.explanations.base import ExplanationBase


class TestPDPTabular(unittest.TestCase):
    def test_1(self):
        set_random_seed()
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = PartialDependenceTabular(training_data=task.train_data, predict_function=predict_function)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()

    def test_2(self):
        set_random_seed()
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = PartialDependenceTabular(training_data=task.train_data, predict_function=predict_function)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())

    def test_3(self):
        set_random_seed()
        task = TabularClassification().train_iris()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = PartialDependenceTabular(training_data=task.train_data, predict_function=predict_function)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = PartialDependenceTabular.load(directory=directory)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())


if __name__ == "__main__":
    unittest.main()
