import numpy as np
import rasterio
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

with rasterio.open("outputs/flood_prediction.tif") as pred_src:
    pred = pred_src.read(1)

with rasterio.open("data/labels/flood_label.tif") as gt_src:
    gt = gt_src.read(1)

# Threshold
pred_bin = (pred > 0.5).astype(np.uint8)
gt = (gt > 0.5).astype(np.uint8)

# Flatten
pred_flat = pred_bin.flatten()
gt_flat = gt.flatten()

acc = accuracy_score(gt_flat, pred_flat)
f1 = f1_score(gt_flat, pred_flat)
iou = jaccard_score(gt_flat, pred_flat)

print("Accuracy:", acc)
print("F1 Score:", f1)
print("IoU:", iou)