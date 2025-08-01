from pathlib import Path
import shutil

import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt

from object_detect import YoloPredictor
from appearance_embed import ReID, EfficientReIDStrategy
from oc_sort import OCSort
from oc_sort import appereance_batch
from camera_motion_compensate import UnifiedCMC




def save_app_matrix_heatmap(app_matrix, trk_ids, filename='app_matrix_heatmap.png', dpi=100):
    plt.figure(figsize=(8, 5))

    im = plt.imshow(app_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label='Cosine Similarity')

    plt.xlabel('Track ID')
    plt.ylabel('Detection Index')
    plt.title('Appearance Similarity Heatmap')

    plt.xticks(ticks=np.arange(len(trk_ids)), labels=trk_ids, rotation=45)
    plt.yticks(ticks=np.arange(app_matrix.shape[0]), labels=np.arange(app_matrix.shape[0]))

    # Annotate each cell with its value
    num_rows, num_cols = app_matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            val = app_matrix[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


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


# load camera motion compensator
cmc = UnifiedCMC(**config["CMC"])

# load ReID strategy
use_efficient_strategy = config["EXP"]["use_efficient_strategy"]
if use_efficient_strategy:
    reid_strategy = EfficientReIDStrategy(**config["EfficientReIDStrategy"])

# load model
predictor = YoloPredictor(**config["YOLO"])  # for object detection
embedder = ReID(**config["ReID"])  # for appearance embedding

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
debug_app_matrix = Path("vis_app_matrix")
for folder in [trk_save_folder, det_save_folder, debug_pred_save_folder, debug_vdir_save_folder,
                debug_lastOb_save_folder, debug_prevCenter_save_folder, debug_firstAssign_save_folder,
                debug_secondAssign_save_folder, debug_thirdAssign_save_folder,
                debug_newlyCreate_save_folder, debug_newlyDelete_save_folder, debug_app_matrix]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()


# load tracker
tracker = OCSort(**config["OCSort"])

np.set_printoptions(suppress=True, precision=3, linewidth=150)
resize_ratio = config["EXP"]["resize_ratio"]
use_cmc = config["EXP"]["use_cmc"]
delayed_update = config["EXP"]["delayed_update"]
num_images = len(img_paths)
print(f"images num: {num_images}")
for i, img_path in enumerate(img_paths):
    # if i > 300:
    #     continue
    
    img_id = int(img_path.stem)
    if img_id == 285:
        print("debug")
    print(f"processing {i+1}/{num_images} img: {img_path}")


    img = cv2.imread(img_path)
    if resize_ratio < 0.9 or resize_ratio > 1.1:
        img = cv2.resize(img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)


    if use_cmc and cmc.img_shape is None:
        cmc.set_img_shape(img.shape[:2])

    # detect(YOLO)
    det_results = predictor.predict(img)
    
    # extract appearance embedding(ReID)
    # selectively pick detections to do appearance embedding(use strategy)
    if use_efficient_strategy:
        target_position = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
        is_target_tracked = True
        selective_det_idx = reid_strategy.select(det_results, target_position, is_target_tracked)
    else:
        selective_det_idx = np.arange(len(det_results))

    # do embedding
    det_feats = np.zeros((len(det_results), embedder.feat_dim), dtype=np.float32)
    def_feats_mask = np.zeros(len(det_results), dtype=bool)  
    for i, (cx, cy, w, h, score) in enumerate(det_results):
        if i in selective_det_idx:
            x1 = max(int(cx - 0.5 * w), 0)
            y1 = max(int(cy - 0.5 * h), 0)
            x2 = min(int(cx + 0.5 * w), img.shape[1])
            y2 = min(int(cy + 0.5 * h), img.shape[0])
            det_feats[i] = embedder.embed(img[y1:y2, x1:x2])
            def_feats_mask[i] = True

    # CMC=camera motion compensation(local to gloabl)
    if use_cmc:
        if not delayed_update:
            _ = cmc.update(img, det_results)
        global_det_results = cmc.local_to_global(det_results)
    else:
        global_det_results = det_results
        


    # track
    debug_mode = True
    if debug_mode:
        rtn_tracks, debug_info = tracker.update(global_det_results, det_feats, def_feats_mask, debug_mode)
    else:
        rtn_tracks = tracker.update(global_det_results, det_feats, def_feats_mask, debug_mode)

    # camera motion compensation(global to local)
    if use_cmc:
        # load tracks coordinate(global)
        global_trks = np.zeros((len(rtn_tracks), 4), dtype=np.float32)
        global_trks_velocity = np.zeros((len(rtn_tracks), 2), dtype=np.float32)
        for i, trk in enumerate(rtn_tracks):
            global_trks[i] = [trk.cx, trk.cy, trk.w, trk.h]
            global_trks_velocity[i] = [trk.vx, trk.vy]
        # convert from global to local
        local_trks = cmc.global_to_local(global_trks)
        for i, trk in enumerate(rtn_tracks):
            trk.cx, trk.cy, trk.w, trk.h = local_trks[i]
        # convert velocity direction from global to local for display 
        M = cmc.cumu_affine_matrix[:2, :2].copy()
        scale_x = np.linalg.norm(M[:, 0])
        scale_y = np.linalg.norm(M[:, 1])
        R = np.zeros((2, 2))
        R[:, 0] = M[:, 0] / scale_x
        R[:, 1] = M[:, 1] / scale_y
        global_trks_velocity_for_display = np.dot(global_trks_velocity, R.T)  # (#trks, 2)
        for i, trk in enumerate(rtn_tracks):
            trk.vx, trk.vy = global_trks_velocity_for_display[i]

    if use_cmc and delayed_update:  # delay update 
        _ = cmc.update(img, det_results)


    # =================================visualization=================================
    # draw tracker result
    img_vis_trk = img.copy()

    # draw camera motion info 
    img_vis_trk = cmc.draw_camera_info(img_vis_trk)

    # draw detect result
    for i, (cx, cy, w, h, score) in enumerate(det_results):
        if i in selective_det_idx:
            embedded = True
        else:
            embedded = False
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), 
                    (0,0,255), 1)
        label = f"{i}: {score:.2f}"
        cv2.putText(img_vis_trk, label, (x2 - 40, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255), 2)
        if embedded:
            cv2.circle(img_vis_trk, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)

    # draw trackers
    scale = 20
    for trk in rtn_tracks:
        x1 = int(trk.cx - 0.5 * trk.w)
        y1 = int(trk.cy - 0.5 * trk.h)
        x2 = int(trk.cx + 0.5 * trk.w)
        y2 = int(trk.cy + 0.5 * trk.h)
        id = int(trk.id)
        less_saturate = not trk.detected
        color = get_color(id, less_saturate)
        cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{id}"
        cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        
        # draw velocity
        cx = int(trk.cx)
        cy = int(trk.cy)
        end_x = int(trk.cx + scale * trk.vx)
        end_y = int(trk.cy + scale * trk.vy)
        cv2.arrowedLine(img_vis_trk, (cx, cy), (end_x, end_y), 
                        color=color, thickness=2, tipLength=0.3)
        
        # Calculate the velocity magnitude
        magnitude = (trk.vx**2 + trk.vy**2)**0.5
        magnitude_label = f"{magnitude:.2f}"

        text_x = cx
        text_y = cy + 15  # shift downward; adjust value as needed
        cv2.putText(img_vis_trk, magnitude_label, (text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # draw similarity
        update_similarity = trk.update_similarity
        label = f" {update_similarity:.2f}"
        cv2.putText(img_vis_trk, label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        


        
        # draw occlusion info
        # occluded = trk.occluded
        # canonical_s = trk.canonical_s
        # canonical_r = trk.canonical_r
        # canonical_w = np.sqrt(canonical_s * canonical_r)
        # canonical_h = canonical_s / (canonical_w + 1e-6)
        # x1 = int(trk.cx - 0.5 * canonical_w)
        # y1 = int(trk.cy - 0.5 * canonical_h)
        # x2 = int(trk.cx + 0.5 * canonical_w)
        # y2 = int(trk.cy + 0.5 * canonical_h)
        # color = (0,255,0) if not occluded else (255,0,0)
        # cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)

    cv2.imwrite(trk_save_folder / img_path.name, img_vis_trk)


    # draw appearance embedding
    
    # get appearnce of existing trackers
    # trk_feats = np.zeros((len(tracker.trackers), embedder.feat_dim), dtype=np.float32)  
    # trk_ids = []
    # for i in range(len(tracker.trackers)):
    #     app = tracker.trackers[i].get_appearance()
    #     if app is None:  # not seen yet due to ReID strategy
    #         app = np.zeros(embedder.feat_dim, dtype=np.float32)
    #     trk_feats[i] = app

    #     trk_ids.append(tracker.trackers[i].id)

    
    # app_matrix = appereance_batch(det_feats, trk_feats)
    # app_vis_save_path = debug_app_matrix / img_path.name
    # save_app_matrix_heatmap(app_matrix, trk_ids, str(app_vis_save_path))



