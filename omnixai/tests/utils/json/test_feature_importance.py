import unittest
import pandas as pd
from omnixai.explanations.tabular.feature_importance import FeatureImportance, ExplanationBase


class TestFeatureImportance(unittest.TestCase):

    def test(self):
        exp = FeatureImportance(mode="classification")
        exp.add(
            instance=pd.DataFrame([["a", "b"], ["c", "d"]], columns=["col 1", "col 2"]),
            target_label=0,
            feature_names=["a", "b", "c"],
            feature_values=[1, 2, 3],
            importance_scores=[0.1, 0.2, 0.3]
        )
        s = exp.to_json()
        self.assertEqual(s, '{"module": "omnixai.explanations.tabular.feature_importance", '
                            '"class": "FeatureImportance", '
                            '"data": {"mode": "classification", '
                            '"explanations": [{"instance": {"col 1": {"0": "a", "1": "c"}, '
                            '"col 2": {"0": "b", "1": "d"}}, "features": ["a", "b", "c"], '
                            '"values": [1, 2, 3], "scores": [0.1, 0.2, 0.3], "target_label": 0}]}}')
        e = ExplanationBase.from_json(s)

        a, b = exp.get_explanations(0), e.get_explanations(0)
        for name in ["features", "values", "scores"]:
            self.assertListEqual(a[name], b[name])
        self.assertEqual(a["target_label"], b["target_label"])

        self.assertListEqual(list(a["instance"].columns), list(b["instance"].columns))
        self.assertListEqual(a["instance"].values.tolist(), b["instance"].values.tolist())


if __name__ == "__main__":
    unittest.main()
