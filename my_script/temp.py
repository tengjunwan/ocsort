from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

info = np.array([
    [175.0, 568.0, 59.4, 125.2, 0.92],
    [259.0, 408.2, 34.9, 71.5, 0.88],
    [343.5, 226.6, 28.8, 28.5, 0.87],
    [287.0, 308.5, 24.0, 47.8, 0.85],
    [343.0, 295.0, 21.5, 39.5, 0.82],
    [61.9, 531.0, 99.7, 93.2, 0.78],
    [244.5, 500.0, 45.4, 123.2, 0.76],
    [354.5, 133.2, 13.8, 18.9, 0.76],
    [203.2, 370.5, 41.9, 90.0, 0.61],
    [280.2, 173.6, 22.0, 34.8, 0.59],
    [402.5, 181.0, 35.8, 20.4, 0.47],
    [485.0, 577.0, 25.2, 63.0, 0.44],
    [258.0, 145.8, 31.0, 16.6, 0.42],
    [219.6, 632.5, 56.2, 15.0, 0.36],
    [373.2, 110.3, 12.5, 17.6, 0.26]
])


img = cv2.imread("imgs/frame_jpg/0000001.jpg")
img_vis_det = img.copy()
for cx, cy, w, h, score in info:
    x1 = int(cx - 0.5 * w)
    y1 = int(cy - 0.5 * h)
    x2 = int(cx + 0.5 * w)
    y2 = int(cy + 0.5 * h)
    cv2.rectangle(img_vis_det, (x1, y1), (x2, y2), 
                    (0,0,255), 2)
    label = f" {score:.2f}"
    cv2.putText(img_vis_det, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,0,255), 1)
cv2.imwrite("temp_result.jpg", img_vis_det)