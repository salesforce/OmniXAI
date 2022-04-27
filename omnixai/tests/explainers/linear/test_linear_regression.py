#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import pandas as pd
from omnixai.utils.misc import set_random_seed
from sklearn.datasets import fetch_california_housing
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import LinearRegression

pd.set_option("display.max_columns", None)


class TestLinearTabular(unittest.TestCase):
    def test_explain(self):
        np.random.seed(1)
        housing = fetch_california_housing()
        df = pd.DataFrame(
            np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
            columns=list(housing.feature_names) + ["target"],
        )
        tabular_data = Tabular(df, target_column="target")
        set_random_seed()
        model = LinearRegression()
        model.fit(tabular_data)

        i = 25
        test_x = tabular_data.iloc(i)
        explanations = model.explain(test_x)
        e = explanations.get_explanations()
        self.assertAlmostEqual(e["coefficients"]["intercept"], 2.072, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
