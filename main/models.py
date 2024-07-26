import torch
import torch.nn as nn
import torch.nn.functional as F
from batchbald_redux import consistent_mc_dropout

class BayesianConvNet(consistent_mc_dropout.BayesianModule):
    '''
    Implementation of a Bayesian convolutional neural network with two convolutional layers and two fully connected layers.
    '''
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
    

class ConvNet(nn.Module):
    '''
    Convolutional neural network with two convolutional layers and two fully connected layers,
    compatible with BackPACK.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.dim_out = 32
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=self.dim_out, kernel_size=5, stride=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(4*4*self.dim_out, self.dim_out),
            nn.GELU(),
            nn.Linear(self.dim_out, 10)
        )

        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    '''
    Multi-layer perceptron with two hidden layers.
    '''

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    '''
    Convolutional neural network with two convolutional layers and two fully connected layers.
    '''

    def __init__(self):
        super(CNN, self).__init__()
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
