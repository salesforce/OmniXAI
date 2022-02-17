#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.cem import CEMOptimizer


class InputData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(nn.Linear(320, 50), nn.ReLU(), nn.Dropout(), nn.Linear(50, 10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def loss(self, x, labels):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return nn.CrossEntropyLoss()(x, labels)


class TestCEM(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_data = torchvision.datasets.MNIST(root="../../data/tmp", train=True, download=True)
        test_data = torchvision.datasets.MNIST(root="../../data/tmp", train=False, download=True)
        train_data.data = train_data.data.numpy()
        test_data.data = test_data.data.numpy()
        self.class_names = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.model = MNISTNet().to(self.device)
        transform = transforms.Compose([transforms.ToTensor()])
        self.transform = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

        self.predict_function = lambda ims: self.model(self.transform(ims).to(self.device)).detach().cpu().numpy()
        self.x_train, self.y_train = Image(train_data.data, batched=True), train_data.targets
        self.x_test, self.y_test = Image(test_data.data, batched=True), test_data.targets
        self.train(learning_rate=1e-3, batch_size=128, num_epochs=10)

    def train(self, learning_rate=1e-3, batch_size=32, num_epochs=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader = DataLoader(
            dataset=InputData(self.transform(self.x_train), self.y_train), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=InputData(self.transform(self.x_test), self.y_test), batch_size=batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                loss = self.model.loss(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % 1000 == 0:
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, total_loss / 2000))
                    total_loss = 0

        correct_pred = {name: 0 for name in self.class_names}
        total_pred = {name: 0 for name in self.class_names}

        self.model.eval()
        for x, y in test_loader:
            images, labels = x.to(device), y.to(device)
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[self.class_names[label]] += 1
                total_pred[self.class_names[label]] += 1

        for name, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[name]
            print("Accuracy for class {} is: {:.1f} %".format(name, accuracy))

    def test(self):
        for idx in range(100):
            x0 = self.transform(self.x_test[idx]).numpy()
            optimizer = CEMOptimizer(x0=x0, target=self.y_test[idx], model=self.model)
            delta = optimizer.pn_optimize()
            if delta is not None:
                fig, axes = plt.subplots(1, 2, squeeze=False)
                plt.sca(axes[0, 0])
                plt.imshow(Image(x0 * 255, batched=True, channel_last=False).to_pil())
                plt.sca(axes[0, 1])
                plt.imshow(Image(delta * 255, batched=True, channel_last=False).to_pil())
                plt.title("PN")
                plt.show()

            delta = optimizer.pp_optimize()
            if delta is not None:
                fig, axes = plt.subplots(1, 2, squeeze=False)
                plt.sca(axes[0, 0])
                plt.imshow(Image(x0 * 255, batched=True, channel_last=False).to_pil())
                plt.sca(axes[0, 1])
                plt.imshow(Image(delta * 255, batched=True, channel_last=False).to_pil())
                plt.title("PP")
                plt.show()


if __name__ == "__main__":
    unittest.main()
