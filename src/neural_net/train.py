import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from neural_net import NeuralNetwork


#hyperparams
batch_size = 16
epochs = 50
lr = 0.001


class myDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def preprocess_data():
    X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
    y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
    X = X.to_numpy()
    y = y.to_numpy()
    N,_ = y.shape
    y =y.reshape((N,1))
    for i in range(len(X[:, 6])):
        X[i, 6] = ord(X[i, 6]) - ord('A')
 
    for i in range(np.shape(X)[1]) :
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    
    real_y = np.zeros(shape=(N, 15))
    
    for i in range(N):
        real_y[i, y[i, 0]] = 1

    X = X.astype(np.float32)
    real_y = real_y.astype(np.float32)


    return X, real_y


def main():
    X_raw, y_raw = preprocess_data()

    #number of samples in data
    N,_ = X_raw.shape

    inputs = torch.Tensor(X_raw)
    labels = torch.Tensor(y_raw)

    print("Inputs", inputs)
    print("Labels", labels)

    # create dataset
    dataset = TensorDataset(inputs,labels)


    #create a split
    val_percent = 0.2
    val_size = int(N * val_percent)

    train_size = N - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #datasets
    print("Train size", train_size)
    print("Val size", val_size)

    print()
    print()
    print()

    # create dataloaders

    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size)

    #instantiate network
    model = NeuralNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #start training loop

    for epoch in range(epochs):

        #training phase
        print("Epoch #", epoch)
        train_loop(train_loader, model, loss_fn, optimizer)
        val_loop(val_loader, model, loss_fn)
            
    print("Done!")

    # model = NeuralNetwork()
    # predictions = model.forward(inputs)


def train_loop(train_loader, model, loss_fn, optimizer):

    total = len(train_loader.dataset)
    for batch_num, (X,y) in enumerate(train_loader):

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def val_loop(val_loader, model, loss_fn):
    test_loss, correct = 0, 0

    val_total_batches = len(val_loader.dataset)
    val_total = len(val_loader.dataset)
            
    with torch.no_grad():
        for batch_num, (X,y) in enumerate(val_loader):
            N,_ = X.shape
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1).reshape(N,1) == y.argmax(1).reshape(N,1)).type(torch.float).sum().item()

    test_loss /= val_total_batches
    correct /= float(val_total)

    print(f"Val Error: \nAccuracy: {(100*correct):>0.1f}%, Avg Val loss: {test_loss:>8f} \n")

# X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
# y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
# X = X.to_numpy()
# y = y.to_numpy()

# for i in range(len(X[:, 6])):
#     X[i, 6] = ord(X[i, 6]) - ord('A')
 
# X = X.astype(np.float32)
# y = y.astype(np.float32)

# X_train = X[:int(0.8 * len(X))]
# y_train = y[:int(0.8 * len(y))]
# X_test = X[int(0.8 * len(X)):]
# y_test = y[int(0.8 * len(y)):]

# train_data = myDataSet(X_train, y_train)
# test_data = myDataSet(X_test, y_test)

# train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# learning_rate = 1e-3

# loss_fn = torch.nn.MSELoss()


# model = NeuralNetwork()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# epochs = 1
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")

if __name__ == '__main__':
    main()