import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.bn1 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Input (32, 32, 1)
        # conv1 => (28, 28, 6) => MaxPool(2, 2) => (14, 14, 6)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # conv2 => (10, 10, 16) => MaxPool(2, 2) => (5, 5, 16)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.conv2(x)
        # x = self.conv2_drop(x)
        x = self.bn1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x) 
        # Flatten (5*5*16)
        x = x.view(-1, self.num_flat_features(x))
        # fc1 => (1, 1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = F.dropout(x, training=self.training)
        # fc2 => (1, 1, 84)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # fc3 => (1, 1, 10)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()
print(net)

