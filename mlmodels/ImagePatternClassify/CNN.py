import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=10, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2) # (-1, 6, 29, 29)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=9, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1( F.relu( self.bn1( self.conv1(x) ) ) )
        x = self.pool2( F.relu( self.bn2( self.conv2(x) ) ) )
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Return class probabilities via a log_softmax function
        return torch.log_softmax(x, dim=1)

