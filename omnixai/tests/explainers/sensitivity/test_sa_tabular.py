#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import pprint
import unittest
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import SensitivityAnalysisTabular
from omnixai.tests.explainers.tasks import TabularRegression


class TestSensitivity(unittest.TestCase):
    def test_1(self):
        set_random_seed()
        task = TabularRegression().train_boston_continuous()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = SensitivityAnalysisTabular(training_data=task.train_data, predict_function=predict_function)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())
        # self.assertAlmostEqual(explanations.get_explanations()["AGE"]["mu"], -0.3059, delta=1e-3)
        # self.assertAlmostEqual(explanations.get_explanations()["B"]["mu"], 0.3111, delta=1e-3)

    def test_2(self):
        set_random_seed()
        task = TabularRegression().train_boston_continuous()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = SensitivityAnalysisTabular(training_data=task.train_data, predict_function=predict_function)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = SensitivityAnalysisTabular.load(directory=directory)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())


if __name__ == "__main__":
    unittest.main()
