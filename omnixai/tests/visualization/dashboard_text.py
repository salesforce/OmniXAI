#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import transformers
from omnixai.data.text import Text
from omnixai.explainers.nlp import NLPExplainer
from omnixai.visualization.dashboard import Dashboard


class TestDashboard(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocess = lambda x: x.values
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
        self.postprocess = lambda outputs: np.array([[s["score"] for s in ss] for ss in outputs])

    def test(self):
        x = Text(
            [
                "What a great movie! if you have no taste.",
                "The Interview was neither that funny nor that witty. "
                "Even if there are words like funny and witty, the overall structure is a negative type.",
            ]
        )
        explainer = NLPExplainer(
            explainers=["shap", "polyjuice"],
            mode="classification",
            model=self.model,
            preprocess=self.preprocess,
            postprocess=self.postprocess,
        )
        local_explanations = explainer.explain(x)
        dashboard = Dashboard(instances=x, local_explanations=local_explanations)
        dashboard.show()


if __name__ == "__main__":
    unittest.main()
