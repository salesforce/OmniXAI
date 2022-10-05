import unittest
import numpy as np
from omnixai.explanations.base import PredictedResults, ExplanationBase


class TestPrediction(unittest.TestCase):

    def test_1(self):
        exp = PredictedResults(np.array([[0.1, 0.2, 0.7], [0.6, 0.1, 0.3]]))
        s = exp.to_json()
        self.assertEqual(s, '{"module": "omnixai.explanations.base", '
                            '"class": "PredictedResults", "data": {"results": '
                            '{"labels": [[2, 1, 0], [0, 2, 1]], "values": [[0.7, 0.2, 0.1], '
                            '[0.6, 0.3, 0.1]]}}}')

        e = ExplanationBase.from_json(s)
        results = e.get_explanations()
        self.assertListEqual(results["labels"][0], [2, 1, 0])
        self.assertListEqual(results["labels"][1], [0, 2, 1])
        self.assertListEqual(results["values"][0], [0.7, 0.2, 0.1])
        self.assertListEqual(results["values"][1], [0.6, 0.3, 0.1])

    def test_2(self):
        exp = PredictedResults(np.array([0.1, 0.2, 0.7]))
        s = exp.to_json()
        e = ExplanationBase.from_json(s)
        results = e.get_explanations()
        self.assertEqual(results["labels"], None)
        self.assertListEqual(results["values"], [0.1, 0.2, 0.7])


if __name__ == "__main__":
    unittest.main()
