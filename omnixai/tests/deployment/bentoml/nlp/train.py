#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import pandas as pd
import torch
import torch.nn as nn

from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer
from omnixai.explainers.nlp import NLPExplainer
from omnixai.deployment.bentoml.omnixai import save_model

from model import TextModel


def train():
    train_data = pd.read_csv('/home/ywz/data/imdb/labeledTrainData.tsv', sep='\t')
    n = int(0.8 * len(train_data))
    x_train = Text(train_data["review"].values[:n])
    y_train = train_data["sentiment"].values[:n].astype(int)
    transform = Word2Id().fit(x_train)

    max_length = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = ["negative", "positive"]

    model = TextModel(
        num_embeddings=transform.vocab_size,
        num_classes=len(class_names)
    ).to(device)

    Trainer(
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-3,
        batch_size=256,
        num_epochs=10,
    ).train(
        model=model,
        loss_func=nn.CrossEntropyLoss(),
        train_x=transform.transform(x_train),
        train_y=y_train,
        padding=True,
        max_length=max_length,
        verbose=True
    )

    def _preprocess(X: Text):
        import numpy as np
        samples = transform.transform(X)
        max_len = 0
        for i in range(len(samples)):
            max_len = max(max_len, len(samples[i]))
        max_len = min(max_len, max_length)
        inputs = np.zeros((len(samples), max_len), dtype=int)
        masks = np.zeros((len(samples), max_len), dtype=np.float32)
        for i in range(len(samples)):
            x = samples[i][:max_len]
            inputs[i, :len(x)] = x
            masks[i, :len(x)] = 1
        return inputs, masks

    def preprocess_func(x):
        import torch
        return tuple(torch.tensor(y).to(device) for y in _preprocess(x))

    def postprocess_func(logits):
        import torch
        return torch.nn.functional.softmax(logits, dim=1)

    explainer = NLPExplainer(
        explainers=["ig", "lime"],
        mode="classification",
        model=model,
        preprocess=preprocess_func,
        postprocess=postprocess_func,
        params={"ig": {"embedding_layer": model.embedding,
                       "id2token": transform.id_to_word}}
    )
    save_model("nlp_explainer", explainer)
    print("Save explainer successfully.")


if __name__ == "__main__":
    train()
