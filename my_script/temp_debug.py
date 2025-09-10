import numpy as np

np.set_printoptions(suppress=True, precision=3, linewidth=150)

from utils import iou_batch
from camera_motion_compensate import UnifiedCMC
import yaml
import cv2

def get_color(idx, less_saturate=False):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    
    if less_saturate:
        # Blend color toward gray (128, 128, 128) to reduce saturation
        gray = 128
        blend_ratio = 0.5  # Adjust this ratio to control how much desaturation is applied
        color = tuple(int(c * (1 - blend_ratio) + gray * blend_ratio) for c in color)
    
    return color

# global track boxes predicted at frame 178
predict_box = """-100.693,-190.471,120.43,219.085
125.048,377.622,84.589,108.781
-46.7685,-195.073,57.9962,93.2046
23.8763,-76.1459,56.5011,94.238
453.953,625.797,85.7784,113.161
-108.911,-280.583,57.3461,71.0184
-19.8027,-33.4979,81.6921,146.712
112.993,255.932,76.0105,85.9304
-110.353,19.0336,94.5712,168.951
-183.399,-229.018,68.7681,23.3935
-299.632,-825.152,68.8778,76.7925
537.107,556.275,31.9187,19.2083
-40.5942,-16.846,65.8579,94.3641
-209.914,-122.694,57.6046,94.7746
-202.448,-216.928,59.5096,94.9927"""

# global detections at frame 178
detection = """-100.59,-191.436,120.963,219.383,0.996094
24.3522,-75.7729,56.5566,95.2003,0.928711
-46.6987,-195.07,58.7706,92.0806,0.912109
-200.789,-215.08,54.8458,89.4138,0.907227
-208.843,-122.163,55.7515,92.4832,0.825684
-19.2912,-33.4446,79.0988,139.681,0.757812
-109.164,-279.095,57.6636,73.9161,0.730469"""

# id corresponding to track boxes
ids = """1
2
3
4
5
6
8
9
10
11
12
13
14
15
16"""

# parse print
predict_box = predict_box.split("\n")
for i in range(len(predict_box)):
    predict_box[i] = [float(j) for j in predict_box[i].split(",")]
predict_box = np.array(predict_box)
print(f"predict_box:\n{predict_box}")

ids = np.array([int(i) for i in ids.split("\n")])

detection = detection.split("\n")
for i in range(len(detection)):
    detection[i] = [float(j) for j in detection[i].split(",")]
detection = np.array(detection)
print(f"detection:\n{detection}")

# test iou function (result: correct)
iou = iou_batch(detection[:,:4], predict_box, buffer_ratioA=0.0, buffer_ratioB=0.2)
print(f"iou:\n{iou}")

# then i want to test CMC 

# cumulative matrix caculated by python code
# cumu_affine_matrix_referecne = np.array([[ 1.257, -0.043, 92.995],
#        [ 0.043,  1.257, 61.849],
#        [ 0.   ,  0.   ,  1.   ]], dtype=np.float32)

# cumulative matrix calculated by Cpp code, it seems OK compared to one by python code
cumu_affine_matrix = np.array([[1.241975771165027, -0.01880523502210651, 98.83028380503087],
                                [0.01880523502210649, 1.241975771165028, 74.65021248816342],
                                [0, 0, 1]], dtype=np.float32)


CONFIG_FILE = './my_script/ocsort_config.yaml'
with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)

# we only want test global_to_local 
cmc = UnifiedCMC(**config["CMC"])
cmc.img_shape = (640, 640)
cmc.cumu_affine_matrix = cumu_affine_matrix


local_detection = cmc.global_to_local(detection)
global_detection = cmc.local_to_global(local_detection)

print(f"local_detection:\n{local_detection}")
print(f"global_detection:\n{global_detection}")

# it seems CMC is alright

# let's try to visualize the detections with proper frame
frame = cv2.imread("test_frame/DJI_20250606154940_0004_V_black_car/00000178.jpg")

vis_frame = frame.copy()
# draw local detection 
for i, (cx, cy, w, h, score) in enumerate(local_detection):
    x1 = int(cx - 0.5 * w)
    y1 = int(cy - 0.5 * h)
    x2 = int(cx + 0.5 * w)
    y2 = int(cy + 0.5 * h)
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), 
                (0,0,255), 1)
    label = f"{i}: {score:.2f}"
    cv2.putText(vis_frame, label, (x2 - 40, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,0,255), 2)

# draw local trackers 
global_trackers = np.concatenate([predict_box, ids.reshape(-1, 1)], axis=1)
local_trackers = cmc.global_to_local(global_trackers)
for i, (cx, cy, w, h, id) in enumerate(local_trackers):
    x1 = int(cx - 0.5 * w)
    y1 = int(cy - 0.5 * h)
    x2 = int(cx + 0.5 * w)
    y2 = int(cy + 0.5 * h)
    id = int(id)
    less_saturate = False
    color = get_color(id, less_saturate)
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{id}"
    cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 2)
    
cv2.imwrite("./debug_vis.jpg", vis_frame)

print("done")


