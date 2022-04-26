import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(10, 25)
        self.relu2 = torch.nn.ELU()
        self.linear3 = torch.nn.Linear(25, 20)
        self.relu4 = torch.nn.ELU()
        self.linear5 = torch.nn.Linear(20, 17)
        self.relu6 = torch.nn.ELU()
        self.linear7 = torch.nn.Linear(17, 15)
        self.soft8 = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.soft8(x)
        return x