import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return x


class MultiTaskLSTM(nn.Module):

    def __init__(self, c_static, time_steps):
        super().__init__()

        self.encoder = CNNEncoder(c_static + 1)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True
        )

        self.flood_head = nn.Conv2d(128, 1, 1)
        self.land_head = nn.Conv2d(128, 1, 1)

    def forward(self, x_static, x_dynamic):

        B, T, H, W = x_dynamic.shape

        seq_features = []

        for t in range(T):

            rain = x_dynamic[:, t:t+1]
            x = torch.cat([x_static, rain], dim=1)

            feat = self.encoder(x)
            feat = feat.mean(dim=[2,3])  # Global pooling

            seq_features.append(feat)

        seq = torch.stack(seq_features, dim=1)

        lstm_out, _ = self.lstm(seq)

        final_feat = lstm_out[:, -1]
        final_feat = final_feat[:, :, None, None]

        flood = self.flood_head(final_feat)
        land  = self.land_head(final_feat)

        return flood, land
