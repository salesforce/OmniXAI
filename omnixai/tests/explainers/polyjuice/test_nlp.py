import unittest
import transformers
import numpy as np

from omnixai.data.text import Text
from omnixai.explainers.nlp.counterfactual.polyjuice import Polyjuice


class TestPolyjuice(unittest.TestCase):

    def setUp(self) -> None:
        try:
            self.model = transformers.pipeline(
                "sentiment-analysis",
                model="/home/ywz/data/models/distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True,
            )
        except:
            self.model = transformers.pipeline(
                "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True
            )

        def _predict(x):
            scores = []
            predictions = self.model(x.values)
            for pred in predictions:
                score = [0.0, 0.0]
                for d in pred:
                    if d['label'] == 'NEGATIVE':
                        score[0] = d['score']
                    else:
                        score[1] = d['score']
                scores.append(score)
            return np.array(scores)

        self.idx2label = {"NEGATIVE": 0, "POSITIVE": 1}
        self.explainer = Polyjuice(
            predict_function=_predict,
            model_path="/home/ywz/data/models/uw-hai/polyjuice"
        )

    def test_explain(self):
        text = Text([
            "it was a fantastic performance!",
            "it was a horrible movie"
        ])
        explanations = self.explainer.explain(text)

        fig = explanations.plotly_plot(index=1)
        fig.show()


if __name__ == "__main__":
    unittest.main()
