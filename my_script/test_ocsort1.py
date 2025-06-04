from pathlib import Path
import shutil

import numpy as np
import cv2
import yaml

from inference_onnx import YoloPredictor
from oc_sort import OCSort

CONFIG_FILE = './my_script/ocsort_config.yaml'
with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)


def get_color(idx, less_saturate=False):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    
    if less_saturate:
        # Blend color toward gray (128, 128, 128) to reduce saturation
        gray = 128
        blend_ratio = 0.5  # Adjust this ratio to control how much desaturation is applied
        color = tuple(int(c * (1 - blend_ratio) + gray * blend_ratio) for c in color)
    
    return color


# load model
predictor = YoloPredictor(**config["YOLO"])

# load imgs
# img_foler = Path("imgs/frame_yuv")
img_foler = Path(config["EXP"]["img_foler"])
img_paths = list(img_foler.glob("*"))
img_paths = sorted(img_paths)

# create vis folder
trk_save_folder = Path("vis_trk")
det_save_folder = Path("vis_det")
debug_pred_save_folder = Path("vis_pred")
debug_vdir_save_folder = Path("vis_vdir")
debug_lastOb_save_folder = Path("vis_lastOb")
debug_prevCenter_save_folder = Path("vis_prevCenter") 
debug_firstAssign_save_folder = Path("vis_firstAssign")
debug_secondAssign_save_folder = Path("vis_secondAssign")
debug_thirdAssign_save_folder = Path("vis_thirdAssign")
debug_newlyCreate_save_folder = Path("vis_newlyCreate")
debug_newlyDelete_save_folder = Path("vis_newlyDelete")
for folder in [trk_save_folder, det_save_folder, debug_pred_save_folder, debug_vdir_save_folder,
                debug_lastOb_save_folder, debug_prevCenter_save_folder, debug_firstAssign_save_folder,
                debug_secondAssign_save_folder, debug_thirdAssign_save_folder,
                debug_newlyCreate_save_folder, debug_newlyDelete_save_folder]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()


# load tracker
tracker = OCSort(**config["OCSort"])

np.set_printoptions(suppress=True, precision=3, linewidth=150)
resize_ratio = config["EXP"]["resize_ratio"]
num_images = len(img_paths)
print(f"images num: {num_images}")
for i, img_path in enumerate(img_paths):
    print(f"processing {i+1}/{num_images} img...")
    img = cv2.imread(img_path)
    if resize_ratio < 0.9 or resize_ratio > 1.1:
        img = cv2.resize(img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)

    # detect
    results = predictor.predict(img)
    
    # track
    debug_mode = True
    if debug_mode:
        online_targets, offline_targets, debug_info = tracker.update(results, debug_mode)
    else:
        online_targets, offline_targets = tracker.update(results, debug_mode)

    # draw tracker result
    img_vis_trk = img.copy()

    # draw detect result
    for cx, cy, w, h, score in results:
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), 
                    (0,0,255), 1)
        label = f" {score:.2f}"
        cv2.putText(img_vis_trk, label, (x2 - 40, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255), 2)

    # draw online trackers
    for cx, cy, w, h, id in online_targets:
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        id = int(id)
        color = get_color(id)
        cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{id}"
        cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        
    # draw temporary offline trackers
    offline_ids = set()
    for cx, cy, w, h, id in offline_targets:
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        id = int(id)
        offline_ids.add(id)
        color = get_color(id, True)
        cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{id}"
        cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)

    if debug_mode:
        scale = 20
        for cx, cy, vx, vy, id in debug_info["v_kalmanfilter"]:
            cx = int(cx)
            cy = int(cy)
            id = int(id)
            offline = id in offline_ids
            color = get_color(id, offline)

            

            end_x = int(cx + scale * vx)
            end_y = int(cy + scale * vy)
            cv2.arrowedLine(img_vis_trk, (cx, cy), (end_x, end_y), 
                            color=color, thickness=2, tipLength=0.3)
            
            # Calculate the velocity magnitude
            magnitude = (vx**2 + vy**2)**0.5
            magnitude_label = f"{magnitude:.2f}"

            # Draw the magnitude label just below the center point
            text_x = cx
            text_y = cy + 15  # shift downward; adjust value as needed
            cv2.putText(img_vis_trk, magnitude_label, (text_x, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=2, lineType=cv2.LINE_AA)
            
        # draw stable shape
        for trk in tracker.trackers:
            stable_w = np.sqrt(trk.canonical_s * trk.canonical_r)
            stable_h = trk.canonical_s / (stable_w + 1e-6)
            shape_deviated = trk.is_deviated
            cx, cy = trk.x.flatten()[:2]
            x1 = int(cx - 0.5 * stable_w)
            y1 = int(cy - 0.5 * stable_h)
            x2 = int(cx + 0.5 * stable_w)
            y2 = int(cy + 0.5 * stable_h)
            color = (0,255,255) if shape_deviated else (0,255,0)
            if shape_deviated:
                cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        
    cv2.imwrite(trk_save_folder / img_path.name, img_vis_trk)
