import unittest
import numpy as np
import pandas as pd
from omnixai.explanations.utils import np_to_json, json_to_np, \
    pd_to_json, json_to_pd


class TestExplanationUtils(unittest.TestCase):

    def test_np_to_json(self):
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        s = np_to_json(x)
        self.assertEqual(s, "[[1, 2, 3, 4], [5, 6, 7, 8]]")

    def test_json_to_np(self):
        s = "[[1, 2, 3, 4], [5, 6, 7, 8]]"
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        y = json_to_np(s)
        self.assertEqual(np.sum(np.abs(x - y)), 0)

    def test_pd_to_json(self):
        df = pd.DataFrame(
            [["a", "b"], ["c", "d"]],
            index=["row 1", "row 2"],
            columns=["col 1", "col 2"],
        )
        s = pd_to_json(df)
        self.assertEqual(s, '{"row 1": {"col 1": "a", "col 2": "b"}, "row 2": {"col 1": "c", "col 2": "d"}}')

    def test_json_to_pd(self):
        s = '{"row 1": {"col 1": "a", "col 2": "b"}, "row 2": {"col 1": "c", "col 2": "d"}}'
        df = json_to_pd(s)
        self.assertListEqual(list(df.columns), ["col 1", "col 2"])
        self.assertListEqual(list(df.index), ["row 1", "row 2"])
        self.assertListEqual(list(df.values[0]), ["a", "b"])
        self.assertListEqual(list(df.values[1]), ["c", "d"])


if __name__ == "__main__":
    unittest.main()
