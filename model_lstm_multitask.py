import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLSTM(nn.Module):
    def __init__(self, static_channels=5, dynamic_channels=24, hidden_dim=32):
        super(MultiTaskLSTM, self).__init__()

        # Encoder for static + one timestep dynamic
        self.encoder = nn.Sequential(
            nn.Conv2d(static_channels + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM over time dimension
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Flood head
        self.flood_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Landslide head
        self.landslide_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_s, x_d):
        """
        x_s: [B, 5, H, W]
        x_d: [B, 24, H, W]
        """

        B, T, H, W = x_d.shape

        encoded_seq = []

        # Loop over time steps
        for t in range(T):
            x_t = x_d[:, t:t+1]  # [B,1,H,W]

            x = torch.cat([x_s, x_t], dim=1)  # [B,6,H,W]
            feat = self.encoder(x)  # [B,32,H,W]

            # Global average pooling to feed LSTM
            pooled = F.adaptive_avg_pool2d(feat, 1).view(B, -1)  # [B,32]

            encoded_seq.append(pooled)

        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B,T,32]

        lstm_out, _ = self.lstm(encoded_seq)  # [B,T,hidden_dim]

        # Use last timestep output
        final_feat = lstm_out[:, -1]  # [B,hidden_dim]

        # Expand back to spatial map
        final_feat = final_feat.view(B, -1, 1, 1)
        final_feat = final_feat.expand(-1, -1, H, W)

        # Heads
        f_logits = self.flood_head(final_feat)
        l_logits = self.landslide_head(final_feat)

        # CLAMP to prevent overflow
        f_logits = torch.clamp(f_logits, -20, 20)
        l_logits = torch.clamp(l_logits, -20, 20)

        return f_logits, l_logits