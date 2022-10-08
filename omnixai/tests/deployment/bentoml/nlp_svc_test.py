#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import torch.nn as nn
import numpy as np
from omnixai.data.text import Text
from omnixai.deployment.bentoml.omnixai import init_service


class TextModel(nn.Module):

    def __init__(self, num_embeddings, num_classes, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = kwargs.get("embedding_size", 50)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.embedding.weight.data.normal_(mean=0.0, std=0.01)

        hidden_size = kwargs.get("hidden_size", 100)
        kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes]

        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.embedding_size, hidden_size, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, num_classes)

    def forward(self, inputs, masks):
        embeddings = self.embedding(inputs)
        x = embeddings * masks.unsqueeze(dim=-1)
        x = x.permute(0, 2, 1)
        x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
        outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(dim=1)
        return outputs


def test():
    svc = init_service(
        model_tag="nlp_explainer:latest",
        task_type="nlp",
        service_name="nlp_explainer"
    )
    for runner in svc.runners:
        runner.init_local()

    x = Text([
        "it was a fantastic performance!",
        "best film ever",
    ])
    predictions = svc.apis["predict"].func(x)
    print(predictions)
    local_explanations = svc.apis["explain"].func(x, {})

    from omnixai.explainers.base import AutoExplainerBase
    from omnixai.visualization.dashboard import Dashboard
    exp = AutoExplainerBase.parse_explanations_from_json(local_explanations)
    dashboard = Dashboard(instances=x, local_explanations=exp)
    dashboard.show()


if __name__ == "__main__":
    test()
