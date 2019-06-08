import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Label: 1 means positive (both images of the same class), 0: negative (different classes)
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNetworkLarge(nn.Module):
    def __init__(self, in_size=(30, 30)):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
 
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )

        n_linear = 64 * (in_size[0] - (2 * 3)) * (in_size[1] - (2 * 3))
        self.fc1 = nn.Sequential(
            nn.Linear(n_linear, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 20)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(40, 2),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        dots = output1 * output2
        diff = output1 - output2

        feats = torch.cat([dots, diff], dim=1)
        preds = F.log_softmax(self.fc2(feats), dim=1)

        return preds