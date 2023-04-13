#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


class TestNLPRequest(unittest.TestCase):

    def test(self):
        data = 'it was a fantastic performance!'

        result = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "text/plain"},
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
