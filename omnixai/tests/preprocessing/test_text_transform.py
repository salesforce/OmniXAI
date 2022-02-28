#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from omnixai.data.text import Text
from omnixai.preprocessing.text import Tfidf, Word2Id


class TestText(unittest.TestCase):
    def test_1(self):
        text = Text(data=["Hello I'm a single sentence", "And another sentence", "And the very very last one"])
        transform = Tfidf()
        transform.fit(text)
        vectors = transform.transform(text)
        print(vectors)
        print(vectors.shape)
        print(transform.get_feature_names())

    def test_2(self):
        text = Text(data=["Hello I'm a single sentence.", "And another sentence.", "And the very very last one."])
        transform = Word2Id()
        transform.fit(text)
        vectors = transform.transform(text)
        print(vectors)
        print(transform.invert(vectors))

        text = Text(data=["Hello I'm a single xxx"])
        print(transform.transform(text))
        print(transform.invert(transform.transform(text)))


if __name__ == "__main__":
    unittest.main()
