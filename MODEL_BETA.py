import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Standard Convolutional Block with 2 layers and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    """Spatially downsamples the input by a factor of 4."""
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


class ConvLSTMCell(nn.Module):
    """A single Convolutional LSTM Cell for preserving spatial dims over time."""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim, 
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class MultiTaskConvLSTM(nn.Module):
    """
    Complete Spatio-Temporal Model.
    Takes static features and dynamic time-series features, 
    and outputs 2D spatial risk maps for flood and landslides.
    """
    def __init__(self, c_static):
        super().__init__()
        
        # 1. Spatial Encoder
        self.encoder = CNNEncoder(in_ch=c_static + 1)
        
        # 2. Temporal Model
        self.conv_lstm = ConvLSTMCell(input_dim=64, hidden_dim=64, kernel_size=3)
        
        # 3. Spatial Decoder (Upsamples H/4, W/4 back to H, W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # 4. Multi-task Heads (1 channel out for each mask)
        self.flood_head = nn.Conv2d(16, 1, kernel_size=1)
        self.land_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x_static, x_dynamic):
        B, T, H, W = x_dynamic.shape

        # Expand static to match Time dim, add Channel dim to dynamic
        x_static_expanded = x_static.unsqueeze(1).expand(-1, T, -1, -1, -1)
        rain = x_dynamic.unsqueeze(2)
        
        # Concat along channel dimension: [B, T, c_static + 1, H, W]
        x = torch.cat([x_static_expanded, rain], dim=2)

        # FOLD Batch and Time for vectorized encoding
        # Using reshape() prevents contiguous memory errors
        x = x.reshape(B * T, -1, H, W)
        encoded = self.encoder(x)
        
        # UNFOLD back to Sequence
        _, C_enc, H_enc, W_enc = encoded.shape
        encoded_seq = encoded.reshape(B, T, C_enc, H_enc, W_enc)

        # Initialize hidden and cell states (new_zeros safely matches device/dtype)
        h = encoded.new_zeros((B, 64, H_enc, W_enc))
        c = encoded.new_zeros((B, 64, H_enc, W_enc))

        # Step through time
        for t in range(T):
            h, c = self.conv_lstm(encoded_seq[:, t, :, :, :], (h, c))

        # Decode the final hidden state
        decoded = self.decoder(h)

        # Multi-task predictions
        flood = self.flood_head(decoded)
        land  = self.land_head(decoded)

        return flood, land


# ==========================================
# Example Usage & Shape Verification
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 4
    TIME_STEPS = 5
    CHANNELS_STATIC = 3  # e.g., elevation, slope, soil type
    HEIGHT = 128
    WIDTH = 128

    # Create dummy data
    # Static data shape: [Batch, Channels, Height, Width]
    dummy_static = torch.randn(BATCH_SIZE, CHANNELS_STATIC, HEIGHT, WIDTH)
    
    # Dynamic data shape: [Batch, Time, Height, Width] (assuming 1 feature like rain)
    dummy_dynamic = torch.randn(BATCH_SIZE, TIME_STEPS, HEIGHT, WIDTH)

    # Initialize model
    model = MultiTaskConvLSTM(c_static=CHANNELS_STATIC)

    # Forward pass
    flood_pred, land_pred = model(dummy_static, dummy_dynamic)

    print(f"Input Static Shape:  {dummy_static.shape}")
    print(f"Input Dynamic Shape: {dummy_dynamic.shape}")
    print("-" * 30)
    print(f"Flood Output Shape:  {flood_pred.shape}")
    print(f"Land Output Shape:   {land_pred.shape}")
