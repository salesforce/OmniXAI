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
from sklearn.datasets import fetch_california_housing
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.specific.decision_tree import TreeRegressor

pd.set_option("display.max_columns", None)


class TestTreeTabular(unittest.TestCase):
    def test_explain(self):
        np.random.seed(1)
        housing = fetch_california_housing()
        df = pd.DataFrame(
            np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
            columns=list(housing.feature_names) + ["target"],
        )
        tabular_data = Tabular(df, target_column="target")
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
