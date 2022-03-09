#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pprint
import xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import ShapTreeTabular

pd.set_option("display.max_columns", None)


class TestShapTreeTabular(unittest.TestCase):
    def test_explain(self):
        np.random.seed(1)
        boston = load_boston()
        tabular_data = Tabular(
            np.concatenate([boston.data, boston.target.reshape((-1, 1))], axis=1),
            feature_columns=list(boston.feature_names) + ["target"],
            categorical_columns=[boston.feature_names[i] for i in [3, 8]],
            target_column="target",
        )

        model = ShapTreeTabular(mode="regression", model=xgboost.XGBRegressor())
        model.fit(tabular_data)

        i = 25
        test_x = tabular_data.iloc(i)
        print(test_x)
        print(model.predict(test_x))
        pprint.pprint(model.explain(test_x).get_explanations())


if __name__ == "__main__":
    unittest.main()
