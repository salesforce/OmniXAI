import os
import unittest
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.deployment.bentoml.omnixai import get, init_service


class TestService(unittest.TestCase):

    def setUp(self) -> None:
        feature_names = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country",
            "label",
        ]
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
        tabular_data = Tabular(
            np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        ).remove_target_column()
        self.test_instances = tabular_data[0:2].to_numpy()

    def test(self):
        model = get("tabular_explainer:latest")
        svc = init_service(model, "tabular_explainer")
        for runner in svc.runners:
            runner.init_local()

        predictions = svc.apis["predict"].func(self.test_instances)
        print(predictions.get_explanations())
        local_explanations = svc.apis["explain"].func(self.test_instances, {})
        print(local_explanations)
        global_explanations = svc.apis["explain_global"].func({})
        print(global_explanations)


if __name__ == "__main__":
    unittest.main()
