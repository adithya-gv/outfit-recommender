import numpy as np
import torch
import os
from torchvision import transforms
import pandas as pd
from ClothingDataset import ClothingDataset
from torch.utils.data import DataLoader, random_split

from cnn import CNN


#hyperparameters
epochs = 10
lr = 0.001
batch_size = 32


#directories
data_dir = 'data/images'
# augmentations 

transforms = transform = transforms.Compose(
    [   
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

#data preprocessing (read from file and then create a dataset class) (only need to call once to create file)
def get_image_list():
    X = pd.read_csv("data/training/articles_subset.csv")[['article_id','cluster']] # you can add more features

    for index, row in X.iterrows():
        name = str(row['article_id'])
        name = '0' + name
        image_name = name[:3] + '/' + name + '.jpg'
        X['article_id'][index] = image_name
        print(X['article_id'][index])
    X.to_csv('data/training/image_label_list.csv', index=False, header=True)
    return X

def main():

    dataset = ClothingDataset(data_dir, transform=transforms)

    val_percent = 0.2
    val_size = int(val_percent * len(dataset))
    train_size = len(dataset) - val_size


    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])
    print("training size:", train_size)
    print("validation size:",val_size)

    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size)

    model = CNN()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #start training loop

    for epoch in range(epochs):

        #training phase
        print("Epoch:", epoch)
        running_loss = train_loop(train_loader, model, loss_fn, optimizer)
        print("Running Loss:", running_loss)
        val_loop(val_loader, model, loss_fn)
        torch.save(model.state_dict(), "data/training/image_model_" + str(epoch) + ".pth")
            
    print("Done!")
    torch.save(model.state_dict(), "data/training/image_model.pth")

    # model = NeuralNetwork()
    # predictions = model.forward(inputs)


def train_loop(train_loader, model, loss_fn, optimizer):

    total = len(train_loader.dataset)
    running_loss = 0.0
    for batch_num, (X,y) in enumerate(train_loader):

        # Compute prediction and loss
        pred = model(X)

        # Backpropagation
        optimizer.zero_grad()
        y = y.type(torch.long)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_num % 10 == 0:
            loss, current = loss.item(), batch_num * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")
    return running_loss


def val_loop(val_loader, model, loss_fn):
    test_loss, correct = 0, 0

    val_total_batches = len(val_loader.dataset)
    val_total = len(val_loader.dataset)
            
    with torch.no_grad():
        for batch_num, (X,y) in enumerate(val_loader):
            N= X.shape[0]
            pred = model(X)
            y = y.type(torch.long)
            # print("Prediction", pred)
            # print("Class pred", pred.argmax(1).reshape(N,1))
            # print("Labels", y.reshape(N,1))
            # print(y.shape)
            test_loss += loss_fn(pred, y).item()
            count = (pred.argmax(1).reshape(N,1) == y.reshape(N,1)).type(torch.float).sum().item()
            correct += count

    test_loss /= val_total_batches
    correct /= float(val_total)

    print(f"Val Error: \nAccuracy: {(100*correct):>0.1f}%, Avg Val loss: {test_loss:>8f} \n")





if __name__ == '__main__':
    main()






