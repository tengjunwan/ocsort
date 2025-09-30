from object_detect import YoloPredictor
from camera_motion_compensate import UnifiedCMC

from pathlib import Path
import shutil

from tqdm import tqdm
import cv2
import numpy as np

from camera_motion_compensate import UnifiedCMC
from object_detect import YoloPredictor
    

predictor = YoloPredictor(onnx_path="my_script/yolov8n_visdrone_add_people_merged_640_640.onnx",
                            conf_threshold=0.25,
                            nms_threshold=0.7)
# cmc = UnifiedCMC(min_features=80, method='orb')
# cmc = UnifiedCMC(min_features=30, method='optflow', process_img_shape=(144, 192), debug_mode=True)
cmc = UnifiedCMC(min_features=30, method='optflow', process_img_shape=(240, 320), debug_mode=True)
np.set_printoptions(precision=3, suppress=True)  # for better numpy.npdarray print

vis_dir = Path("vis_cmc")
if vis_dir.exists():
    shutil.rmtree(vis_dir)
vis_dir.mkdir()


img_dir = Path("test_imgs/DJI_20250912_gray_suv_seg_1")
img_paths = sorted(list(img_dir.glob("*.jpg")))
img_num = len(img_paths)
for j, img_path in tqdm(enumerate(img_paths), total=img_num):
    img_id = int(img_path.stem)
    if img_id == 225:
        print("debug")
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (640, 640))

    dets = predictor.predict(img)  # for removing moving foregrounds
    curr_affine_matrix = cmc.update(img, dets)

    img_vis = cmc.vis_img
    if img_vis is None:
        continue


    # global_dets = cmc.local_to_global(dets)
    # local_dets = cmc.global_to_local(global_dets)

    # img_vis = img.copy()
    # # draw global cooridate 
    # for i in range(len(dets)):
    #     cx, cy, w, h, score, cls_id = local_dets[i]
    #     x1 = int(cx - 0.5 * w)
    #     y1 = int(cy - 0.5 * h)
    #     x2 = int(cx + 0.5 * w)
    #     y2 = int(cy + 0.5 * h)
    #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), 
    #                     (0,0,255), 1)
    #     g_cx, g_cy, g_w, g_h, score, cls_id = global_dets[i]
    #     label = f"({g_cx:.1f}, {g_cy:.1f}): {score:.2f}"
    #     cv2.putText(img_vis, label, (x2 - 80, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, (0,0,255), 2)
    # cmc.draw_camera_info(img_vis)
    
    cv2.imwrite(vis_dir / img_path.name, img_vis)
        