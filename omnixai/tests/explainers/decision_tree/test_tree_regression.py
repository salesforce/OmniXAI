#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pprint
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.specific.decision_tree import TreeRegressor

pd.set_option("display.max_columns", None)


class TestTreeTabular(unittest.TestCase):
    def test_explain(self):
        np.random.seed(1)
        boston = load_boston()
        tabular_data = Tabular(
            np.concatenate([boston.data, boston.target.reshape((-1, 1))], axis=1),
            feature_columns=list(boston.feature_names) + ["target"],
            categorical_columns=[boston.feature_names[i] for i in [3, 8]],
            target_column="target",
        )
        np.random.seed(1)
        model = TreeRegressor()
        model.fit(tabular_data, max_depth=4)

        i = 25
        test_x = tabular_data.iloc(i)
        print(test_x)
        print(model.predict(test_x))
        explanations = model.explain(test_x)
        for e in explanations.get_explanations()["path"]:
            pprint.pprint([p["text"] for p in e])


if __name__ == "__main__":
    unittest.main()
