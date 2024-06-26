import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    '''
    Convolutional neural network with two convolutional layers and two fully connected layers.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.dim_out = 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=self.dim_out, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*self.dim_out, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.dim_out)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
