#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import requests
import numpy as np
from omnixai.data.tabular import Tabular
from requests_toolbelt.multipart.encoder import MultipartEncoder


class TestTabularRequest(unittest.TestCase):

    def setUp(self) -> None:
        # Load the dataset
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
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../datasets")
        tabular_data = Tabular(
            np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )
        print(tabular_data[0].to_numpy())

    def test(self):
        data = '["39", "State-gov", "77516", "Bachelors", "13", "Never-married", ' \
               '"Adm-clerical", "Not-in-family", "White", "Male", "2174", "0", "40", "United-States"]'

        result = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "application/json"},
            data=data
        ).text
        print(result)

        m = MultipartEncoder(
            fields={
                "data": data,
                "params": '{"lime": {"y": [0]}}',
            }
        )
        result = requests.post(
            "http://0.0.0.0:3000/explain",
            headers={"Content-Type": m.content_type},
            data=m
        ).text
        print(result)


if __name__ == "__main__":
    unittest.main()
