from torch import nn


class LeNet5(nn.Module):
    def __init__(self, n_classes: int = 13):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, n_classes)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class CNN(nn.Module):
    def __init__(self, n_classes: int = 13):
        super(CNN, self).__init__()
        self.conv0 = nn.Sequential(
                        nn.Conv2d(1, 16, (5,5), (2,2), (0,0)),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(16, 32, (3,3), (2,2), (0,0)),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),)
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 16, (3,3), (1,1), (0,0)),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),)
        self.out = nn.Linear(16 * 3 * 3, n_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
