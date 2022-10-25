import unittest
from omnixai.explainers.prediction import PredictionAnalyzer
from omnixai.tests.explainers.tasks import TabularRegression
from omnixai.explanations.base import ExplanationBase


class TestResidual(unittest.TestCase):

    def test_regression_metric(self):
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))

        explainer = PredictionAnalyzer(
            predict_function=predict_function,
            test_data=task.test_data,
            test_targets=task.test_targets,
            mode="regression"
        )
        explanations = explainer._regression_residual()
        explanations.plotly_plot()
        explanations.plot()

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()


if __name__ == "__main__":
    unittest.main()
