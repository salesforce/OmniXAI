#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from omnixai.utils.misc import ProgressBar


class SamplingLayer(nn.Module):
    def __init__(self, tau, k):
        super().__init__()
        self.tau = tau
        self.k = k

    def forward(self, logits):
        """
        :param logits: The outputs of the L2X model with shape `(batch_size, d)`.
        """
        d = logits.shape[1]
        batch_size = logits.shape[0]
        outputs = torch.unsqueeze(logits, dim=1)

        uniform = torch.clamp(torch.rand(size=(batch_size, self.k, d), device=logits.device), min=1e-4, max=0.9999)
        gumbel = -torch.log(-torch.log(uniform))
        noisy_logits = (gumbel + outputs) / self.tau
        weights = torch.softmax(noisy_logits, dim=-1)
        weights = torch.max(weights, dim=1)[0]
        threshold = torch.unsqueeze(torch.topk(logits, min(self.k, d), sorted=True)[0][:, -1], dim=-1)
        selections = torch.ge(logits, threshold).float()
        return weights, selections


class L2XModel(nn.Module):
    def __init__(self, selection_model, prediction_model, tau, k):
        super().__init__()
        self.selection_model = selection_model
        self.prediction_model = prediction_model
        self.tau = tau
        self.k = k
        # Post-processing for logits, e.g., upsampling
        postprocess = getattr(selection_model, "postprocess", None)
        if callable(postprocess):
            self.postprocess = selection_model.postprocess
        else:
            self.postprocess = lambda x: x

    def forward(self, *inputs):
        # p(S|X)
        logits = self.selection_model(*inputs)
        weights, _ = SamplingLayer(self.tau, self.k)(logits)
        weights = self.postprocess(weights)
        # q(X_S)
        outputs = self.prediction_model(*inputs, weights)
        return outputs

    def explain(self, *inputs):
        self.eval()
        logits = self.selection_model(*inputs)
        weights, selections = SamplingLayer(self.tau, self.k)(logits)
        weights = self.postprocess(weights)
        selections = self.postprocess(selections)
        return weights, selections


class Trainer:
    def __init__(self, optimizer_class, learning_rate, batch_size, num_epochs):
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(
        self,
        model,
        loss_func,
        train_x,
        train_y,
        valid_x=None,
        valid_y=None,
        padding=False,
        max_length=256,
        verbose=True,
    ):
        device = next(model.parameters()).device
        optimizer = self.optimizer_class(model.parameters(), lr=self.learning_rate)
        train_data = self._data_loader(train_x, train_y, shuffle=True, padding=padding, max_length=max_length)
        bar = ProgressBar(self.num_epochs) if verbose else None

        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for i, data in enumerate(train_data):
                inputs = [x.to(device) for x in data]
                loss = loss_func(model(*inputs[:-1]), inputs[-1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data
            if verbose:
                bar.print(epoch + 1, prefix="", suffix="Complete, Loss {:.4f}".format(total_loss / len(train_data)))

        if valid_x is not None and valid_y is not None:
            model.eval()
            total_loss = 0
            valid_data = self._data_loader(valid_x, valid_y, shuffle=True, padding=padding, max_length=max_length)
            for i, data in enumerate(valid_data):
                inputs = [x.to(device) for x in data]
                loss = loss_func(model(*inputs[:-1]), inputs[-1])
                total_loss += loss.data
            if verbose:
                print("Validation loss: {}".format(total_loss / len(valid_data)))

    def _data_loader(self, x, y, shuffle, padding, max_length):
        if not padding:
            loader = DataLoader(
                dataset=TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=self.batch_size, shuffle=shuffle
            )
        else:
            loader = DataLoader(
                dataset=InputData(x, y, max_length),
                batch_size=self.batch_size,
                collate_fn=InputData.collate_func,
                shuffle=shuffle,
            )
        return loader


class InputData(Dataset):
    def __init__(self, x, y, max_length=None):
        self.x = x
        self.y = y
        self.max_length = max_length

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.max_length is None:
            return self.x[index], self.y[index]
        else:
            return self.x[index][: self.max_length], self.y[index]

    @staticmethod
    def collate_func(samples):
        batch_size = len(samples)
        max_len = 1
        for i in range(batch_size):
            max_len = max(max_len, len(samples[i][0]))

        values, masks = [], []
        for i in range(batch_size):
            length = min(len(samples[i][0]), max_len)
            masks.append([1] * length + [0] * (max_len - length))
            v = samples[i][0][:length] + [0] * (max_len - length)
            values.append(v)
        values = torch.from_numpy(np.array(values, dtype=int))
        masks = torch.from_numpy(np.array(masks, dtype=np.float32))
        targets = torch.LongTensor([samples[i][1] for i in range(batch_size)])
        return values, masks, targets
