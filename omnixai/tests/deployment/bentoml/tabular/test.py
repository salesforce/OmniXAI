#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.deployment.bentoml.omnixai import init_service


def test():
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
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../datasets")
    tabular_data = Tabular(
        np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str),
        feature_columns=feature_names,
        categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
        target_column="label",
    ).remove_target_column()
    test_instances = tabular_data[0:2].to_numpy()

    svc = init_service(
        model_tag="tabular_explainer:latest",
        task_type="tabular",
        service_name="tabular_explainer"
    )
    for runner in svc.runners:
        runner.init_local()

    predictions = svc.apis["predict"].func(test_instances)
    print(predictions)
    local_explanations = svc.apis["explain"].func(test_instances, {})
    print(local_explanations)


if __name__ == "__main__":
    test()
