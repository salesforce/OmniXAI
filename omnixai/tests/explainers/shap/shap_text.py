#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import transformers

from omnixai.data.text import Text
from omnixai.explainers.nlp import ShapText


class TestText(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.model = transformers.pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True,
            )
        except:
            self.model = transformers.pipeline(
                "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True
            )

    def test_explain(self):
        x = Text("What a great movie! if you have no taste.")
        explainer = ShapText(model=self.model)
        explanations = explainer.explain(x)
        explanations.plot()


if __name__ == "__main__":
    unittest.main()
