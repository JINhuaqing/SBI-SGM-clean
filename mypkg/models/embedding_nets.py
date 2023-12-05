import torch
import torch.nn as nn 
import torch.nn.functional as F 


class SummaryNet(nn.Module):
    def __init__(self, num_in_fs=5440, num_out_fs=10):
        super().__init__()
        # 2D convolutional layer
        self.fc1 = nn.Linear(in_features=num_in_fs, out_features=1600)
        self.fc2 = nn.Linear(in_features=1600, out_features=400)
        self.fc3 = nn.Linear(in_features=400, out_features=num_out_fs)

    def forward(self, x):
        #x = x.view(-1, 1, 32, 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x