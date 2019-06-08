import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, n_c_in, n_c_out):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(n_c_in, n_c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_c_out),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(n_c_out, n_c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_c_out),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(n_c_out, n_c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.b1(x)
        r1 = out1  # no residual since input is different size
        out2 = self.b2(r1)
        r2 = r1 + out2
        out3 = self.b3(r2)
        r3 = r2 + out3
        return r3


class FeatNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Encoder(3, 32)
        self.enc2 = Encoder(32, 64)
        self.enc3 = Encoder(64, 64)
        self.enc4 = Encoder(64, 64)
        self.enc5 = Encoder(64, 128)
        self.enc6 = Encoder(128, 128)

    def forward(self, x):
        e1 = self.enc1(x)
        e1_pooled = F.max_pool2d(e1, 2, stride=2)
        e2 = self.enc2(e1_pooled)
        e2_pooled = F.max_pool2d(e2, 2, stride=2)
        e3 = self.enc3(e2_pooled)
        e3_pooled = F.max_pool2d(e3, 2, stride=2)
        e4 = self.enc4(e3_pooled)
        e4_pooled = F.max_pool2d(e4, 2, stride=2)
        e5 = self.enc5(e4_pooled)
        e5_pooled = F.max_pool2d(e5, 2, stride=2)
        e6 = self.enc6(e5_pooled)

        return e6


class SiameseNet(nn.Module):
    def __init__(self, input_side: int = 256) -> None:
        super().__init__()

        # calculate the size of the final feature map
        assert input_side % 32 == 0
        feat_size = 128 * (input_side // 32) ** 2

        self.feature_extractor = FeatNet()
        self.proj = nn.Sequential(
            torch.nn.Linear(feat_size, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),

            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(128),
        )

    def forward_once(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.shape[0], -1)
        return self.proj(feats)

    def forward(self, input1, input2):
        feat1 = self.forward_once(input1)
        feat2 = self.forward_once(input2)

        return feat1, feat2
