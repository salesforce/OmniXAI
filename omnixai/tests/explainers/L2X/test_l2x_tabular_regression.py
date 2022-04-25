#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pprint
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular.agnostic.L2X.l2x import L2XTabular
from omnixai.tests.explainers.tasks import TabularRegression


class TestLimeTabular(unittest.TestCase):
    def test_explain(self):
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = L2XTabular(training_data=task.train_data, predict_function=predict_function, mode="regression")

        set_random_seed()
        i = 25
        test_x = task.test_data.iloc(i)
        explanations = explainer.explain(test_x)
        for e in explanations.get_explanations():
            print(e["instance"])
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))


if __name__ == "__main__":
    unittest.main()
