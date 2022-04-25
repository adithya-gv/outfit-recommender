import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(10, 100)
        self.relu2 = torch.nn.LeakyReLU()
        self.linear3 = torch.nn.Linear(100, 50)
        self.relu4 = torch.nn.LeakyReLU()
        self.linear5 = torch.nn.Linear(50, 13)
        self.soft6 = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.soft6(x)
        return x