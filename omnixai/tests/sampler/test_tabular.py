#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.sampler.tabular import Sampler


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        feature_names = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country",
            "label",
        ]
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
        self.tabular_data = Tabular(
            np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )

    def _check_categorical(self, samples):
        df_a = self.tabular_data.to_pd(copy=False)
        df_b = samples.to_pd(copy=False)
        va = Sampler._get_categorical_values(df_a, self.tabular_data.categorical_columns)
        vb = Sampler._get_categorical_values(df_b, self.tabular_data.categorical_columns)
        self.assertEqual(len(va), len(vb))
        for key in va.keys():
            self.assertEqual(len(va[key]), len(vb[key]))

    def test_subsample(self):
        samples = Sampler.subsample(self.tabular_data, fraction=0.1, random_state=0)
        self._check_categorical(samples)

    def test_undersample(self):
        samples = Sampler.undersample(self.tabular_data, random_state=0)
        self._check_categorical(samples)

        df = samples.to_pd()
        for label in df[self.tabular_data.target_column].unique():
            split = df[df[self.tabular_data.target_column] == label]
            print((label, len(split)))

    def test_oversample(self):
        samples = Sampler.oversample(self.tabular_data, random_state=0)
        self._check_categorical(samples)

        df = samples.to_pd()
        for label in df[self.tabular_data.target_column].unique():
            split = df[df[self.tabular_data.target_column] == label]
            print((label, len(split)))


if __name__ == "__main__":
    unittest.main()
