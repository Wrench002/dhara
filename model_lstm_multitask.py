import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. ConvLSTM Cell — LSTM that operates on full spatial maps
# ─────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Spatially-aware LSTM cell. Hidden states are [B, hidden_dim, H, W],
    so spatial structure is preserved across every timestep.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        # All 4 gates fused into a single conv for efficiency
        self.gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        x : [B, input_dim,  H, W]
        h : [B, hidden_dim, H, W]
        c : [B, hidden_dim, H, W]
        """
        combined = torch.cat([x, h], dim=1)          # [B, input+hidden, H, W]
        i, f, g, o = self.gates(combined).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B: int, H: int, W: int, device: torch.device):
        return (
            torch.zeros(B, self.hidden_dim, H, W, device=device),
            torch.zeros(B, self.hidden_dim, H, W, device=device),
        )


# ─────────────────────────────────────────────────────────────
# 2. ConvLSTM — wraps the cell to process a full sequence
# ─────────────────────────────────────────────────────────────

class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : [B, T, C, H, W]
        returns: [B, T, hidden_dim, H, W]
        """
        B, T, C, H, W = x_seq.shape
        h, c = self.cell.init_hidden(B, H, W, x_seq.device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)             # [B, T, hidden_dim, H, W]


# ─────────────────────────────────────────────────────────────
# 3. Multi-Scale Encoder — dilated convs for multi-scale context
# ─────────────────────────────────────────────────────────────

class MultiScaleEncoder(nn.Module):
    """
    Three parallel dilated conv branches capture features at different
    receptive field sizes simultaneously, without losing spatial resolution.
    Dilation 1 = local terrain, 2 = neighbourhood, 4 = regional drainage/slope.
    """
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        mid = out_channels // 2

        def branch(dilation):
            return nn.Sequential(
                nn.Conv2d(in_channels, mid, kernel_size=3,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )

        self.branch1 = branch(dilation=1)
        self.branch2 = branch(dilation=2)
        self.branch3 = branch(dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        ], dim=1))


# ─────────────────────────────────────────────────────────────
# 4. Main Model
# ─────────────────────────────────────────────────────────────

class MultiTaskConvLSTM(nn.Module):
    """
    Multi-task model for flood and landslide hazard susceptibility mapping.

    Key fixes over the original MultiTaskLSTM:
      - ConvLSTM replaces global-avg-pool → LSTM → expand. Spatial structure
        is preserved throughout — every pixel gets its own LSTM trajectory.
      - Multi-scale dilated encoder captures local + regional terrain context.
      - BatchNorm throughout for stable training (no logit clamping needed).
      - Learned refinement conv after LSTM instead of the expand() broadcast.
      - Both heads operate on spatially-rich [B, hidden_dim, H, W] features.

    Args:
        static_channels  : number of static input channels (DEM, slope, soil…)
        dynamic_channels : number of dynamic timesteps
        encoder_dim      : output channels of the multi-scale encoder
        hidden_dim       : ConvLSTM hidden channels (also input to task heads)
    """
    def __init__(
        self,
        static_channels: int = 5,
        dynamic_channels: int = 24,
        encoder_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Encodes [static | one dynamic timestep] → spatial feature map
        self.encoder = MultiScaleEncoder(
            in_channels=static_channels + 1,
            out_channels=encoder_dim,
        )

        # Processes the full [B, T, encoder_dim, H, W] sequence spatially
        self.conv_lstm = ConvLSTM(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
        )

        # Adds spatial expressiveness after the LSTM
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Task-specific heads
        self.flood_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.landslide_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_s: torch.Tensor, x_d: torch.Tensor):
        """
        x_s : [B, static_channels, H, W]  — static features (DEM, slope, soil…)
        x_d : [B, T, H, W]                — dynamic features over T timesteps

        Returns
        -------
        f_logits : [B, 1, H, W]  — flood susceptibility logits
        l_logits : [B, 1, H, W]  — landslide susceptibility logits

        Apply torch.sigmoid() at inference for probability maps.
        """
        B, T, H, W = x_d.shape

        # Encode each timestep while keeping full [H, W] spatial resolution
        encoded_seq = []
        for t in range(T):
            x_t = x_d[:, t:t+1]                       # [B, 1, H, W]
            x   = torch.cat([x_s, x_t], dim=1)         # [B, static+1, H, W]
            encoded_seq.append(self.encoder(x))         # [B, encoder_dim, H, W]

        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B, T, encoder_dim, H, W]

        # ConvLSTM — hidden state propagates spatially through time
        lstm_out   = self.conv_lstm(encoded_seq)        # [B, T, hidden_dim, H, W]
        final_feat = lstm_out[:, -1]                    # [B, hidden_dim, H, W]
        final_feat = self.refine(final_feat)            # [B, hidden_dim, H, W]

        f_logits = self.flood_head(final_feat)          # [B, 1, H, W]
        l_logits = self.landslide_head(final_feat)      # [B, 1, H, W]

        return f_logits, l_logits


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W = 2, 64, 64
    x_s = torch.randn(B, 5,  H, W)
    x_d = torch.randn(B, 24, H, W)

    model = MultiTaskConvLSTM(
        static_channels=5,
        dynamic_channels=24,
        encoder_dim=32,
        hidden_dim=64,
    )

    f_out, l_out = model(x_s, x_d)
    print(f"Flood logits:     {f_out.shape}")    # [2, 1, 64, 64]
    print(f"Landslide logits: {l_out.shape}")    # [2, 1, 64, 64]

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
