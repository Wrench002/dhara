import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from dataset import NPZDataset
from model_lstm_multitask import MultiTaskLSTM

# ---------------------------
# Device (CPU for stability)
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# Dataset
# ---------------------------
dataset = NPZDataset(config.TRAIN_TILES_DIR)
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

print(f"Total training tiles: {len(dataset)}")

# ---------------------------
# Model
# ---------------------------
model = MultiTaskLSTM().to(device)

pos_weight = torch.tensor([5.0])  # Increase if flood very rare
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

EPOCHS = 15

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for x_s, x_d, y in loader:

        x_s = x_s.to(device)
        x_d = x_d.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        f_logits, l_logits = model(x_s, x_d)

        loss_f = criterion(f_logits, y[:, 0:1])
        loss_l = criterion(l_logits, y[:, 1:2])
        loss = loss_f + loss_l

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

print("Training complete.")
torch.save(model.state_dict(), "model.pth")
print("Model saved.")