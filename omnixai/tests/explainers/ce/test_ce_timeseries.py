import os
import unittest
import numpy as np
from omnixai.utils.misc import set_random_seed
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries.counterfactual.ce import CounterfactualExplainer


class TestShapTimeseries(unittest.TestCase):

    def setUp(self) -> None:
        set_random_seed()
        x = np.random.randn(200, 5)
        x[:, 0] = np.array(range(x.shape[0])) * 0.1
        y = np.random.randn(3, 5)
        y[:, 0] = np.array(range(y.shape[0])) * 0.1

        self.training_data = Timeseries(
            data=x,
            timestamps=list(range(x.shape[0])),
            variable_names=list('x' * (i + 1) for i in range(x.shape[1]))
        )
        self.test_data = Timeseries(
            data=y,
            timestamps=list(range(y.shape[0])),
            variable_names=list('x' * (i + 1) for i in range(x.shape[1]))
        )
        self.predict_function = lambda _x: np.sum(_x.values[:, 0])

    def test(self):
        set_random_seed()
        explainer = CounterfactualExplainer(
            training_data=self.training_data,
            predict_function=self.predict_function,
            threshold=0.2
        )
        explanations = explainer.explain(self.test_data)
        cf = explanations.get_explanations(index=0)["counterfactual"]["x"].values
        self.assertEqual(cf[0], 0.0)
        self.assertAlmostEqual(cf[1], 0.049, delta=1e-3)
        self.assertAlmostEqual(cf[2], 0.149, delta=1e-3)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = CounterfactualExplainer.load(directory=directory)
        explanations = explainer.explain(self.test_data)
        cf = explanations.get_explanations(index=0)["counterfactual"]["x"].values
        self.assertEqual(cf[0], 0.0)
        self.assertAlmostEqual(cf[1], 0.049, delta=1e-3)
        self.assertAlmostEqual(cf[2], 0.149, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
