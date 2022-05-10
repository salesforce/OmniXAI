#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Morris sensitivity analysis for tabular data
"""
from typing import Callable
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.sensitivity import SensitivityExplanation


class SensitivityAnalysisTabular(TabularExplainer):
    """
    Morris sensitivity analysis for tabular data based on the SALib.
    If using this explainer, please cite the package: https://github.com/SALib/SALib.
    This explainer only supports continuous-valued features.
    """

    explanation_type = "global"
    alias = ["sa", "sensitivity"]

    def __init__(self, training_data: Tabular, predict_function: Callable, **kwargs):
        """
        :param training_data: The data used to initialize the explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            The outputs of the ``predict_function`` should be a batch of estimated values, e.g.,
            class probabilities are not supported.
        """
        if "mode" in kwargs:
            kwargs.pop("mode")
        super().__init__(training_data=training_data, predict_function=predict_function, mode="regression", **kwargs)
        assert len(self.categorical_columns) == 0, "The sensitivity analysis only supports continuous-valued features"
        assert len(self.feature_columns) == self.data.shape[1], (
            "The number of continuous-valued features doesn't fit the data dim. " "It is probably caused by a bug."
        )

    def _build_problem(self):
        """
        Builds the problem for Morris analysis.

        :return: The problem information, e.g., "num_vars", "names" and "bounds".
        :rtype: Dict
        """

        def _bound(x):
            min_val, max_val = min(x), max(x)
            if abs(min_val - max_val) < 1e-3:
                min_val -= 1e-3
                max_val += 1e-3
            return min_val, max_val

        bounds = [_bound(self.data[:, i]) for i in range(len(self.feature_columns))]
        problem = {"num_vars": len(self.feature_columns), "names": self.feature_columns, "bounds": bounds}
        return problem

    def explain(self, **kwargs) -> SensitivityExplanation:
        """
        Generates sensitivity analysis explanations.

        :param kwargs: Additional parameters, e.g., ``nsamples`` --
            the number of samples in Morris sampling.
        :return: The generated global explanations.
        """
        nsamples = kwargs.get("nsamples", 1024)
        explanations = SensitivityExplanation()

        problem = self._build_problem()
        samples = morris_sampler.sample(problem, nsamples)
        r = morris.analyze(problem, samples, self.predict_fn(samples))
        for name, mu, mu_star, sigma, mu_star_conf in zip(
            r["names"], r["mu"], r["mu_star"], r["sigma"], r["mu_star_conf"]
        ):
            explanations.add(feature_name=name, mu=mu, mu_star=mu_star, sigma=sigma, mu_star_conf=mu_star_conf)
        return explanations
