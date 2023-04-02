import torch
import torch.nn as nn
import torch.nn.functional as F

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Feature merging layers
        self.merge1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.merge2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Output layers for score, geometry, and angle
        self.score = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.geometry = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)
        self.angle = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Feature merging
        x = F.relu(self.merge1(torch.cat([x, x], dim=1)))
        x = F.relu(self.merge2(x))

        # Output layers
        score = torch.sigmoid(self.score(x))
        geometry = torch.sigmoid(self.geometry(x)) * 512
        angle = (torch.sigmoid(self.angle(x)) - 0.5) * np.pi/2

        return score, geometry, angle
