from pathlib import Path
from tqdm import tqdm
import shutil

import cv2
import numpy as np

from object_detect import YoloPredictor
from camera_motion_compensate import UnifiedCMC

gray_img_dir = Path("test_imgs/gray")
gray_img_paths = sorted(gray_img_dir.glob("*.jpg"))
gray_img_num = len(gray_img_paths)

bgr_img_dir = Path("test_imgs/yuv")
bgr_img_paths = sorted(bgr_img_dir.glob("*.jpg"))
bgr_img_num = len(bgr_img_paths)

assert gray_img_num == bgr_img_num, "something wrong"
np.set_printoptions(precision=3, suppress=True)
# YOLO
vis_dir = Path("vis_cmc")
if vis_dir.exists():
    shutil.rmtree(vis_dir)
vis_dir.mkdir()
predictor = YoloPredictor(onnx_path="my_script/yolov8n_visdrone_2.onnx")
cmc = UnifiedCMC(30, "optflow", 1.0)
for i in tqdm(range(gray_img_num), total=gray_img_num):
    if i+1 == 428:
        print("debug")
    gray = cv2.imread(gray_img_paths[i], cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(bgr_img_paths[i])

    cmc.set_img_shape(bgr.shape[:2])

    dets = predictor.predict(bgr)
    curr_affine_matrix = cmc.update(gray, dets)

    print(f"curr_affine_matrix:\n{curr_affine_matrix}")
    print(f"cumu_affine_matrix:\n{cmc.cumu_affine_matrix}")

    global_dets = cmc.local_to_global(dets)
    local_dets = cmc.global_to_local(global_dets)

    img_vis = bgr.copy()
    # draw global cooridate 
    for j in range(len(dets)):
        cx, cy, w, h, score = local_dets[j]
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), 
                        (0,0,255), 1)
        g_cx, g_cy, g_w, g_h, score = global_dets[j]
        label = f"({g_cx:.1f}, {g_cy:.1f}): {score:.2f}"
        cv2.putText(img_vis, label, (x2 - 80, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255), 2)
    cmc.draw_camera_info(img_vis)
    
    cv2.imwrite(vis_dir / bgr_img_paths[i].name, img_vis)

print("done")