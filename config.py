import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute pathing for all directories
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(DATA_DIR, "static")
DYNAMIC_DIR = os.path.join(DATA_DIR, "dynamic")
LABEL_DIR = os.path.join(DATA_DIR, "labels")

TILES_DIR = os.path.join(BASE_DIR, "tiles")
TRAIN_TILES_DIR = "tiles"
VAL_TILES_DIR = os.path.join(TILES_DIR, "val")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure all folders exist
for folder in [TRAIN_TILES_DIR, VAL_TILES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Parameters
TILE_SIZE = 128
TIME_STEPS = 24     
BATCH_SIZE = 8
EPOCHS = 20         # Balanced for your 1-hour remaining window
C_STATIC = 5