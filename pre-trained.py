import math

import numpy as np
import torch
from torch import nn
import random

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(1200, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(1, 40, 30)
        x = self.flatten(x)
        return self.network(x)


def pre_train(epochs, optimizer, loss_fn):
    model.train()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        for i in range(5):
            data = np.load('data' + str(i + 1) + '.npy')
            avg = 0.0
            for j in range(len(data)):
                x = torch.tensor(data[j, :-1].astype(np.float32))
                x.to('cuda')
                y = torch.tensor(data[j, -1].astype(np.int64))
                y.to('cuda')

                pred = model(x).flatten()
                # print(pred)
                # print(y)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg += loss.item()
            avg /= len(data)
            print("Average loss: {}".format(avg))


model = PolicyNetwork()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss().to('cuda')
epochs = 60
pre_train(epochs, optimizer, loss_fn)
torch.save(model.state_dict(), 'pre_trained model.pth')