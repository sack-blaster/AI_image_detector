# Load pre-trained ResNet-18 model and modify the final layer for binary classification

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        # Load the pre-trained ResNet-18 model with default weights
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer to output 2 classes (binary classification)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet18(x)