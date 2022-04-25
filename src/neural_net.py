import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class myDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(10, 100)
        self.relu2 = torch.nn.LeakyReLU()
        self.linear3 = torch.nn.Linear(100, 50)
        self.relu4 = torch.nn.LeakyReLU()
        self.linear5 = torch.nn.Linear(50, 1)
        self.soft6 = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.soft6(x)
        return x
 
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
X = X.to_numpy()
y = y.to_numpy()

for i in range(len(X[:, 6])):
    X[i, 6] = ord(X[i, 6]) - ord('A')
 
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

train_data = myDataSet(X_train, y_train)
test_data = myDataSet(X_test, y_test)

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

learning_rate = 1e-3

loss_fn = torch.nn.MSELoss()


model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")