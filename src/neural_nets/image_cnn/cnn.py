from xml.dom import INVALID_CHARACTER_ERR
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8,32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),
        
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64 ,64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128,128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(8*8*128,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,16),
            nn.Softmax()
        )

    def forward(self, images):

        return self.network(images)