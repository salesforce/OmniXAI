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
from omnixai.explainers.tabular import GlobalShapTabular
from omnixai.tests.explainers.tasks import TabularClassification


class TestShapTabular(unittest.TestCase):
    def test(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))

        set_random_seed()
        explainer = GlobalShapTabular(
            training_data=task.train_data,
            predict_function=predict_function,
            ignored_features=None,
            nsamples=100
        )
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())


if __name__ == "__main__":
    unittest.main()
