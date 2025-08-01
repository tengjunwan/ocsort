from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

with open("my_script/output.txt", "r") as f:
    content = f.readlines()


elements = content[0].split(",")
elements = [float(i) for i in elements]

elements = np.array(elements)
elements = elements.reshape(1, 3, 128, 128)
# denormalize
mean = np.array([123.6750, 116.2800, 103.5300], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array([58.3950, 57.1200, 57.3750], dtype=np.float32).reshape(1, 3, 1, 1)

elements = elements * std + mean


elements = np.round(elements).astype(np.uint8)
elements = elements[0].transpose(1, 2, 0)  # H,, W, C
elements = elements[:, :, ::-1]  # RGB to BGR

cv2.imwrite("temp.jpg", elements)

print("done")

