import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn

from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp import NLPExplainer
from omnixai.visualization.dashboard import Dashboard


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


class TestDashboard(unittest.TestCase):

    def setUp(self) -> None:
        # Load the training and test datasets
        train_data = pd.read_csv('/home/ywz/data/imdb/labeledTrainData.tsv', sep='\t')
        n = int(0.8 * len(train_data))
        x_train = Text(train_data["review"].values[:n])
        y_train = train_data["sentiment"].values[:n].astype(int)
        x_test = Text(train_data["review"].values[n:])
        y_test = train_data["sentiment"].values[n:].astype(int)
        # The transform for converting words/tokens to IDs
        transform = Word2Id().fit(x_train)

        max_length = 256
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ["negative", "positive"]

        def preprocess(X: Text):
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

        model = TextModel(
            num_embeddings=transform.vocab_size,
            num_classes=len(self.class_names)
        ).to(device)

        Trainer(
            optimizer_class=torch.optim.AdamW,
            learning_rate=1e-3,
            batch_size=128,
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

        model.eval()
        data = transform.transform(x_test)
        data_loader = DataLoader(
            dataset=InputData(data, [0] * len(data), max_length),
            batch_size=32,
            collate_fn=InputData.collate_func,
            shuffle=False
        )
        outputs = []
        for inputs in data_loader:
            value, mask, target = inputs
            y = model(value.to(device), mask.to(device))
            outputs.append(y.detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        predictions = np.argmax(outputs, axis=1)
        print('Test accuracy: {}'.format(
            sklearn.metrics.f1_score(y_test, predictions, average='binary')))

        # The preprocessing function
        preprocess_func = lambda x: tuple(torch.tensor(y).to(device) for y in preprocess(x))
        # The postprocessing function
        postprocess_func = lambda logits: torch.nn.functional.softmax(logits, dim=1)
        # Initialize a NLPExplainer
        self.explainer = NLPExplainer(
            explainers=["ig", "lime", "polyjuice"],
            mode="classification",
            model=model,
            preprocess=preprocess_func,
            postprocess=postprocess_func,
            params={"ig": {"embedding_layer": model.embedding,
                           "id2token": transform.id_to_word}}
        )

    def test_explain(self):
        x = Text([
            "What a great movie! if you have no taste.",
            "it was a fantastic performance!",
            "best film ever",
            "such a great show!",
            "it was a horrible movie",
            "i've never watched something as bad"
        ])
        # Generates explanations
        local_explanations = self.explainer.explain(x)
        # Launch a dashboard for visualization
        dashboard = Dashboard(
            instances=x,
            local_explanations=local_explanations,
            class_names=self.class_names
        )
        dashboard.show()


if __name__ == "__main__":
    unittest.main()
