#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import requests


class TestVisionRequest(unittest.TestCase):

    def setUp(self) -> None:
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/images/")

    def test(self):
        result = requests.post(
            "http://0.0.0.0:3000/predict",
            files={"upload_file": open(os.path.join(self.directory, 'dog_cat.png'), 'rb')},
            headers={"content-type": "multipart/form-data"}
        ).text
        print(result)


if __name__ == "__main__":
    unittest.main()
