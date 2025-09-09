import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_class=10, feat_dim=128):
        super(SimpleCNN, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, feat_dim)  # feature vector h
        self.fc2 = nn.Linear(feat_dim, num_class)   # classification head

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        h = F.relu(self.fc1(x))   # features
        logits = self.fc2(h)

        if return_features:
            return h  # return representation for SimCLR
        return logits


def get_model(num_classes=10, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_class=num_classes).to(device)
    return model
