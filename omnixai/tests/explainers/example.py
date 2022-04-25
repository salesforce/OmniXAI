#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omnixai.explainers.tabular import LimeTabular
from omnixai.explainers.tabular import ShapTabular
from omnixai.explainers.tabular import MACEExplainer
from omnixai.explainers.tabular import PartialDependenceTabular
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestExample(unittest.TestCase):
    def test_explain(self):
        base_folder = os.path.dirname(os.path.abspath(__file__))
        task = TabularClassification(base_folder).train_adult()
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        class_names = task.transform.class_names

        # Setup different explainers
        lime_explainer = LimeTabular(training_data=task.train_data, predict_function=predict_function, kernel_width=3)
        shap_explainer = ShapTabular(training_data=task.train_data, predict_function=predict_function, nsamples=100)
        cf_explainer = MACEExplainer(
            training_data=task.train_data,
            predict_function=predict_function,
            ignored_features=["Sex", "Race", "Relationship", "Capital Loss"],
        )
        pdp_explainer = PartialDependenceTabular(training_data=task.train_data, predict_function=predict_function)

        np.random.seed(1)
        i = 1653
        test_x = task.test_data.iloc([i, i + 1])
        test_x.data.at[i, "Occupation"] = "Unknown"

        # Show explanations
        explanations = lime_explainer.explain(test_x)
        explanations.plot(class_names=class_names, title="LIME")
        plt.show()

        explanations = shap_explainer.explain(test_x)
        explanations.plot(class_names=class_names)
        plt.show()

        scores = predict_function(test_x)
        desired_labels = 1.0 - np.argmax(scores, axis=1)
        explanations = cf_explainer.explain(test_x, desired_labels)
        explanations.plot(class_names=class_names)
        plt.show()

        explanations = pdp_explainer.explain()
        explanations.plot(
            features=[
                "Age",
                "Education-Num",
                "Capital Gain",
                "Capital Loss",
                "Hours per week",
                "Education",
                "Marital Status",
                "Occupation",
                "Workclass",
            ],
            class_names=class_names,
        )
        plt.show()


if __name__ == "__main__":
    unittest.main()
