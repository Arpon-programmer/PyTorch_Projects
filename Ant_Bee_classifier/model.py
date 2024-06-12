import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # Adjusted kernel size
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)  # Adjusted kernel size
        self.fc1 = nn.Linear(16 * 56 * 56, 120)  # Adjusted input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # Adjusted kernel size and stride for pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # Adjusted kernel size and stride for pooling
        x = x.view(-1, 16 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x