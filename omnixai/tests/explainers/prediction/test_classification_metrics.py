import unittest
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from omnixai.explainers.prediction import PredictionAnalyzer
from omnixai.explanations.base import ExplanationBase


class TestClassificationMetrics(unittest.TestCase):

    def test_classification_metric(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target

        random_state = np.random.RandomState(0)
        n_samples, n_features = x.shape
        x = np.c_[x, random_state.randn(n_samples, 20 * n_features)]

        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=0.5, random_state=0)
        classifier = svm.SVC(kernel="linear", probability=True, random_state=0)
        classifier.fit(x_train, y_train)

        explainer = PredictionAnalyzer(
            mode="classification",
            predict_function=lambda z: classifier.predict_proba(z),
            test_data=x_test,
            test_targets=y_test
        )
        explanations = explainer._metric()
        print(explanations.get_explanations())
        explanations.plotly_plot()
        explanations.plot(class_names=["a", "b", "c"])

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()


if __name__ == "__main__":
    unittest.main()
