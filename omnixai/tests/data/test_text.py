#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from omnixai.data.text import Text


class TestText(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test(self):
        text = Text(data=["Hello I'm a single sentence", "And another sentence", "And the very very last one"])
        tokens = text.to_tokens()

        self.assertEqual(len(text), 3)
        self.assertEqual(text[0].to_str(), "Hello I'm a single sentence")
        self.assertCountEqual(tokens[0], ["hello", "i", "'m", "a", "single", "sentence"])
        self.assertCountEqual(tokens[1], ["and", "another", "sentence"])
        self.assertCountEqual(tokens[2], ["and", "the", "very", "very", "last", "one"])


if __name__ == "__main__":
    unittest.main()
