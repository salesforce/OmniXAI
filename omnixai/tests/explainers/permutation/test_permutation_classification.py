#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import PermutationImportance
from omnixai.tests.explainers.tasks import TabularClassification


class TestPermutation(unittest.TestCase):
    def test(self):
        set_random_seed()
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        explainer = PermutationImportance(training_data=task.train_data, predict_function=predict_function)
        explanations = explainer.explain(X=task.test_data, y=task.test_targets)
        explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()
