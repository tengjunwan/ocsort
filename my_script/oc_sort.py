from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# from kalmanfilterbox import KalmanFilterBoxTracker
# from kalmanfilterboxnomatrix import KalmanFilterBoxTrackerNoMatrix as KalmanFilterBoxTracker
from kalmanfilterboxnomatrix_new import KalmanFilterBoxTrackerNoMatrix as KalmanFilterBoxTracker


USE_MY_HUNGARIAN = True
if USE_MY_HUNGARIAN:
    from hungarian_algorithm import hungarian_rect

def cxcywh2xyxy(boxes):
    x1y1x2y2 = np.zeros_like(boxes)
    x1y1x2y2[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    x1y1x2y2[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    x1y1x2y2[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    x1y1x2y2[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return x1y1x2y2


def iou_batch(bboxes1, bboxes2):  # 
    """
    From SORT: Computes IOU between two bboxes in the form [cx,cy,w,h]
        bboxes1: detections, shape = (#dets, 4).
        bboxes2: trackers, shape = (#trks, 4).
    """
    # [cx, cy, w, h] -> [x1, y2, x2, y2]
    bboxes1 = cxcywh2xyxy(bboxes1)
    bboxes2 = cxcywh2xyxy(bboxes2)

    bboxes2 = np.expand_dims(bboxes2, 0)  # (1, #trks, 4)
    bboxes1 = np.expand_dims(bboxes1, 1)  # (#dets, 1, 4)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h  
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return o  # (#dets, #trks)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]  # (#trks, 2, 1)
    CX1 = dets[:,0]  # (#dets)
    CY1 = dets[:,1]  # (#dets)
    CX2 = tracks[:,0]  # (#trks, 1)
    CY2 = tracks[:,1]  # (#trks, 1)
    dx = CX1 - CX2  # (#trks, #dets)
    dy = CY1 - CY2  # (#trks, #dets)
    norm = np.sqrt(dx**2 + dy**2) + 1e-6  # (#trks, #dets)
    dx = dx / norm  # (#trks, #dets)
    dy = dy / norm  # (#trks, #dets)
    return dx, dy # size: num_track x num_det


def linear_assignment(cost_matrix):
    if USE_MY_HUNGARIAN:
        total_cost, matching = hungarian_rect(cost_matrix)
    else:
        x, y = linear_sum_assignment(cost_matrix)
        matching = np.array(list(zip(x, y)))
    return matching

def oc_associate(detections, trackers, iou_threshold, velocities, previous_obs, valid_mask, vdc_weight):  
    """
    Associates detections with trackers based on IoU and optional velocity difference cost.
 
    Args:
        detections (numpy.ndarray): Detection results from a detector (e.g., YOLO),
            shape = (num_detections, 5), where each detection is (cx, cy, w, h, score).
        trackers (numpy.ndarray): Predicted locations from a tracker (e.g., KalmanFilter),
            shape = (num_trackers, 4), where each tracker is (cx, cy, w, h).
        iou_threshold (float): Minimum IoU required to associate a detection with a tracker.
        velocities (numpy.ndarray): Velocity vectors predicted by KalmanFilter,
            shape = (num_trackers, 2), where each vector is (vx, vy).
        previous_obs (numpy.ndarray): Previous observations used to compute velocity differences,
            shape = (num_trackers, 2), (center_x, center_y) per tracker.
        valid_mask (numpy.ndarray): Valid previous observations mask,
            shape = (num_trackers,), 1 means valid, 0 means invalid.
        vdc_weight (float): Weighting factor for velocity direction difference in the matching cost.
 
    Returns:
        matches (numpy.ndarray): Matched detection-tracker pairs,
            shape = (num_matches, 2), each row is (detection_idx, tracker_idx).
        unmatched_detections (numpy.ndarray): Indices of unmatched detections,
            shape = (num_unmatched_detections,).
        unmatched_trackers (numpy.ndarray): Indices of unmatched trackers,
            shape = (num_unmatched_trackers,).
    """  
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    # part 1: calculate v_directioin reward matrix(#dets, #trks)
    X, Y = speed_direction_batch(detections, previous_obs)  # (#trks, #dets)
    inertia_X, inertia_Y = velocities[:,0], velocities[:,1]  # (#trks,)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)  # (#trks, #dets)
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)  # (#trks, #dets)
    diff_angle_cos = inertia_X * X + inertia_Y * Y # (#trks, #dets), cosθ
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)  # θ ∈ [0, pi]
    diff_angle_reward = (np.pi /2.0 - np.abs(diff_angle)) / np.pi  # diff_angle_reward ∈ [-0.5, 0.5]
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)  # (#dets, #trks)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # (#trks, #dets)
    # zero out invalid track rows which have not valid previous_obs
    angle_diff_cost = (valid_mask * diff_angle_reward) * vdc_weight  # (#trks, #dets)
    angle_diff_cost = angle_diff_cost.T  # (#dets, #trks)
    angle_diff_cost = angle_diff_cost * scores  # weighted by scores

    iou_matrix = iou_batch(detections[:, :4], trackers)  # (#dets, #trks), reward matrix

    if min(iou_matrix.shape) > 0:
        matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matching with low iou
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches) == 0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


@dataclass
class Track:
    cx: float
    cy: float
    w: float
    h: float
    id: int
    detected: bool
    vx: float
    vy: float
    occluded: bool
    canonical_s: float
    canonical_r: float



class OCSort(object):
    """
    Observation-Centric SORT (OC-SORT) tracker.

    This class manages multiple object tracks using Kalman filters. It handles
    object association, creation, update, and deletion. OC-SORT improves tracking
    under occlusion by using observation-centric re-updates and optional velocity-based
    matching.

    Args:
        det_thresh (float): 
            Detection confidence threshold. Detections below this are ignored 
            in the first association stage.
        max_age (int): 
            Maximum number of consecutive frames a tracker is allowed to miss 
            before it is removed. Default: 30.
        min_hits (int): 
            Minimum number of successful updates before a tracker is considered 
            valid and included in the output. Default: 3.
        iou_threshold (float): 
            IOU threshold used for assigning detections to existing trackers. Default: 0.3.
        delta_t (int): 
            Time gap (in frames) for estimating object velocity or doing ORU (observation-centric re-update). Default: 3.
        asso_func (str): 
            Name of the association function to use ('iou', 'giou', 'diou', etc). Default: 'iou'.
        inertia (float): 
            Inertia factor for blending motion direction into the association cost. Range: [0, 1]. Default: 0.2.
        use_byte (bool): 
            If True, use BYTE-style secondary matching with low-confidence detections. Default: False.
        low_det_thresh(float):
            low score thresh used in BYTE-style secondary matching. Default: 0.2.
        v_inertia(float):
            to prevent speed estimation noise by partial detection, use EMA to update vx, vy and vs. Default: 0.0

    Attributes:
        trackers (List[KalmanBoxTracker]): 
            List of currently active trackers.
        frame_count (int): 
            Total number of frames processed.
    """
    
    def __init__(self, det_thresh=0.7, max_age=60, min_hits=3, 
        iou_threshold=0.3, delta_t=3, inertia=0.2, use_byte=True, low_det_thresh=0.2, max_track_num=40, **kwargs):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = iou_batch
        self.inertia = inertia
        self.use_byte = use_byte
        self.low_det_thresh = low_det_thresh
        self.max_track_num = max_track_num
        KalmanFilterBoxTracker.count = 0

    def update(self, yolo_dets, debug_mode=False):
        """
        Args:
            yolo_dets (numpy.ndarray): 
                shape (#det, 5), 5 means cx, cy, w, h, score; 
        return:
            yolo_dets_with_id (numpy.ndarray):
                shape (#det, 6), 6 means cx, cy, w, h, score, id;
        """
        if debug_mode:
            debug_info = {
                "predict_bbox": [],  # (#trks, 5), cx, cy, w, h, id
                "v_directions": [],  # (#trks, 5), cx, cy, vx(normalized), vy(normalized), id
                "v_kalmanfilter": [],  # (trks, 5), cx, cy, vx, vy, id
                "last_observed_zs": [],  # (#trks, 5), cx, cy, w, h, id
                "previous_obs": [],  # (#trks, 3), cx, cy, id
                "first_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "second_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "third_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "newly_created": [], # (#num, 5), cx, cy, w, h, id
                "newly_deleted": [], # (#num, 5), cx, cy, w, h, id
            }
        self.frame_count += 1

        if yolo_dets is None or len(yolo_dets) == 0:  # (#det, 5)
            yolo_dets = yolo_dets.reshape(-1, 5)  # (0, 5)
            
            # if debug_mode:
            #     return self._return_trackers(), debug_info
            # else:
            #     return self._return_trackers()
            
        # filter detections by score
        scores = yolo_dets[:, 4]  # (#det)
        high_conf_mask = scores > self.det_thresh
        mid_conf_mask = (scores > self.low_det_thresh) & ~high_conf_mask  # (0.1 < score <= det_thresh)

        dets_high = yolo_dets[high_conf_mask]      # (#dets, 5), Used for primary matching
        dets_second = yolo_dets[mid_conf_mask]     # (#dets, 5), Used for BYTE-style secondary matching

        # get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)  # (#trks, 4), cx, cy, w, h
        for i in range(len(self.trackers)):
            trks[i] = self.trackers[i].predict()
            if debug_mode:
                cx, cy, w, h = trks[i]
                id = self.trackers[i].id
                debug_info["predict_bbox"].append(np.array([cx, cy, w, h, id], dtype=np.float32))


        # get velocity directions from existing trackers
        v_directions = np.zeros((len(self.trackers), 2), dtype=np.float32)
        for i in range(len(self.trackers)):
            v_directions[i] = self.trackers[i].v_direction  # (vx, vy)
            if debug_mode:
                vx, vy = v_directions[i]
                cx, cy, w, h, id = self.trackers[i].get_state_with_id()
                debug_info["v_directions"].append(np.array([cx, cy, vx, vy, id], dtype=np.float32))

        # get last observation for OCR mentioned in OCSORT paper for secondary matching
        last_observed_zs= np.zeros((len(self.trackers), 4), dtype=np.float32)
        for i in range(len(self.trackers)):
            last_z = self.trackers[i].get_last_observed_z()  
            if last_z is None:
                last_observed_zs[i] = np.array([-10, -10, 0, 0])  # won't overlap with any det
            else:
                last_observed_zs[i] = last_z  # (cx, cy, w, h)
            if debug_mode:
                cx, cy, w, h = last_observed_zs[i]
                id = self.trackers[i].id
                debug_info["last_observed_zs"].append(np.array([cx, cy, w, h, id], 
                                                               dtype=np.float32))
        
        # get previous obersavetion for estimating velocity directions
        previous_obs = np.zeros((len(self.trackers), 2), dtype=np.float32)
        valid_mask = np.ones(len(self.trackers), dtype=np.float32)
        for i in range(len(self.trackers)):
            prev_z = self.trackers[i].get_previous_obs_for_v_direction()  # (cx, cy, s, r)
            if prev_z is None:
                previous_obs[i] = np.array([0, 0], dtype=np.float32)  
                valid_mask[i] = 0
            else:
                previous_obs[i] = prev_z.flatten()[:2]  # (cx, cy)

            if debug_mode:
                cx, cy = previous_obs[i]
                id = self.trackers[i].id
                debug_info["previous_obs"].append(np.array([cx, cy, id], dtype=np.float32))

        if debug_mode:
            for i in range(len(self.trackers)):
                cx = self.trackers[i].x[0, 0]
                cy = self.trackers[i].x[1, 0]
                vx_kf = self.trackers[i].x[4, 0]
                vy_kf = self.trackers[i].x[5, 0]
                id = self.trackers[i].id
                debug_info["v_kalmanfilter"].append(np.array([cx, cy, vx_kf, vy_kf, id], dtype=np.float32))

        # first round of ocsort (OCM): iou + v (trackers + high score detections)
        matched, unmatched_dets, unmatched_trks = oc_associate(
            dets_high, trks, self.iou_threshold, v_directions, previous_obs, 
            valid_mask, self.inertia)
        for det_idx, trk_idx in matched:   # update trackers by matched detection immediately
            self.trackers[trk_idx].update(dets_high[det_idx, :4])

            if debug_mode:
                cx, cy, w, h = dets_high[det_idx, :4]
                id = self.trackers[trk_idx].id
                debug_info["first_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))
        
        # second round of bytetrack: iou (optional: left trackers + mid score detections)
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            left_trks = trks[unmatched_trks]
            byte_iou_matrix = iou_batch(dets_second[:, :4], left_trks)  # (#dets_sec, #trks_left)
            byte_matched = linear_assignment(-byte_iou_matrix)
            # filter out matching with low iou
            to_remove_trk_idx = []
            for det_idx_in_sec, trk_idx_in_left in byte_matched:
                if byte_iou_matrix[det_idx_in_sec, trk_idx_in_left] \
                    < self.iou_threshold:
                    continue
                trk_idx = unmatched_trks[trk_idx_in_left]
                self.trackers[trk_idx].update(dets_second[det_idx_in_sec, :4])  # update
                to_remove_trk_idx.append(trk_idx)  # remove unmatched trackers id

                if debug_mode:
                    cx, cy, w, h = dets_second[det_idx_in_sec, :4]  
                    id = self.trackers[trk_idx].id
                    debug_info["second_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))

            unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_idx))

        # third round of ocsort(OCR): iou (last observation of left trackers + left high score detections)
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_trks = last_observed_zs[unmatched_trks]
            left_dets_high = dets_high[unmatched_dets]
            ocr_iou = iou_batch(left_dets_high[:, :4], left_trks) 
            ocr_matched = linear_assignment(-ocr_iou)
            to_remove_det_idx = []
            to_remove_trk_idx = []
            for det_idx_in_left, trk_idx_in_left in ocr_matched:
                if ocr_iou[det_idx_in_left, trk_idx_in_left] < self.iou_threshold:
                    continue
                det_idx = unmatched_dets[det_idx_in_left]
                trk_idx = unmatched_trks[trk_idx_in_left]
                self.trackers[trk_idx].update(dets_high[det_idx, :4])  # update

                if debug_mode:
                    cx, cy, w, h = dets_high[det_idx, :4]
                    id = self.trackers[trk_idx].id
                    debug_info["third_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))

                to_remove_det_idx.append(det_idx)
                to_remove_trk_idx.append(trk_idx)
            unmatched_dets = np.setdiff1d(unmatched_dets, np.array([to_remove_det_idx]))
            unmatched_trks = np.setdiff1d(unmatched_trks, np.array([to_remove_trk_idx]))

        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].update(None)  # update unmatched trackers by None(no measurement)
        
        
        # manage trackers
        num_trackers = len(self.trackers)
        for i in range(num_trackers):
            trk_idx = num_trackers - 1 - i  # loop backwards
            trk_obj = self.trackers[trk_idx]
            # remove dead tracker
            if trk_obj.consecutive_missed_frames > self.max_age:
                self.trackers.pop(trk_idx) 
                if debug_mode:
                    cx, cy, w, h, id = trk_obj.get_state_with_id()
                    debug_info["newly_deleted"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))
                continue
        
        # create new trackers for unmatched high score detections
        for det_idx in unmatched_dets:
            cx, cy, w, h = dets_high[det_idx, :4]
            if len(self.trackers) > self.max_track_num:  # prevent too many trackers
                continue
            self.trackers.append(KalmanFilterBoxTracker(cx, cy, w, h, self.delta_t))
            if debug_mode:
                cx, cy, w, h = dets_high[det_idx, :4]
                id = self.trackers[-1].id
                debug_info["newly_created"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))
        

        if debug_mode:
            return self._return_trackers(), debug_info
        else:
            return self._return_trackers()
        
    def _return_trackers(self):
        rtn_trackers = []
        for trk_obj in self.trackers:
            cx, cy, w, h, id = trk_obj.get_state_with_id()
            id = int(id)
            detected = trk_obj.consecutive_missed_frames == 0
            vx, vy = trk_obj.x.flatten()[4:6]
            is_occluded  = trk_obj.is_occluded
            canonical_s = trk_obj.canonical_s
            canonical_r = trk_obj.canonical_r
            rtn_tracker = Track(cx, cy, w, h, id, detected, vx, vy, is_occluded, canonical_s, canonical_r)
            rtn_trackers.append(rtn_tracker)
        return rtn_trackers


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    import cv2
    import yaml

    from inference_onnx import YoloPredictor

    CONFIG_FILE = './my_script/ocsort_config.yaml'
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)


    # def get_color(idx):
        # idx = idx * 3
        # color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        # return color
    
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
        if resize_ratio < 0.9:
            img = cv2.resize(img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)

        # detect
        detect_results = predictor.predict(img)
        
        # track
        debug_mode = True
        if debug_mode:
            rtn_tracks, debug_info = tracker.update(detect_results, debug_mode)
        else:
            rtn_tracks = tracker.update(detect_results, debug_mode)

        # draw tracker result
        img_vis_trk = img.copy()

        # draw detect result
        for cx, cy, w, h, score in detect_results:
            x1 = int(cx - 0.5 * w)
            y1 = int(cy - 0.5 * h)


        # draw trackers
        for trk in rtn_tracks:
            x1 = int(trk.cx - 0.5 * trk.w)
            y1 = int(trk.cy - 0.5 * trk.h)
            x2 = int(trk.cx + 0.5 * trk.w)
            y2 = int(trk.cy + 0.5 * trk.h)
            less_saturate = not trk.detected
            color = get_color(trk.id, less_saturate)
            cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{trk.id}"
            cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
        # # draw online trackers
        # for cx, cy, w, h, id in online_targets:
        #     x1 = int(cx - 0.5 * w)
        #     y1 = int(cy - 0.5 * h)
        #     x2 = int(cx + 0.5 * w)
        #     y2 = int(cy + 0.5 * h)
        #     id = int(id)
        #     color = get_color(id)
        #     cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        #     label = f"ID:{id}"
        #     cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 0.5, color, 2)
            
        # # draw temporary offline trackers
        # offline_ids = set()
        # for cx, cy, w, h, id in offline_targets:
        #     x1 = int(cx - 0.5 * w)
        #     y1 = int(cy - 0.5 * h)
        #     x2 = int(cx + 0.5 * w)
        #     y2 = int(cy + 0.5 * h)
        #     id = int(id)
        #     offline_ids.add(id)
        #     color = get_color(id, True)
        #     cv2.rectangle(img_vis_trk, (x1, y1), (x2, y2), color, 2)
        #     label = f"ID:{id}"
        #     cv2.putText(img_vis_trk, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 0.5, color, 2)
            
        # draw detect result
        img_vis_det = img.copy()
        for cx, cy, w, h, score in detect_results:
            x1 = int(cx - 0.5 * w)
            y1 = int(cy - 0.5 * h)
            x2 = int(cx + 0.5 * w)
            y2 = int(cy + 0.5 * h)
            cv2.rectangle(img_vis_det, (x1, y1), (x2, y2), 
                        (0,0,255), 2)
            label = f" {score:.2f}"
            cv2.putText(img_vis_det, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,0,255), 1)
        cv2.imwrite(det_save_folder / img_path.name, img_vis_det)
            
        if debug_mode:
            scale = 50
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
            
        cv2.imwrite(trk_save_folder / img_path.name, img_vis_trk)

        
        
        if debug_mode:
            # draw predictions
            img_vis_pred = img.copy() 
            for cx, cy, w, h, id in debug_info["predict_bbox"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_pred, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_pred, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_pred_save_folder / img_path.name, img_vis_pred)

            # draw v directions
            img_vis_vdir = img.copy()
            for cx, cy, vx, vy, id in debug_info["v_directions"]:
                scale = 20
                cx = int(cx)
                cy = int(cy)
                end_x = int(cx + scale * vx)
                end_y = int(cy + scale * vy)
                id = int(id)
                color = get_color(id)
                cv2.arrowedLine(img_vis_vdir, (cx, cy), (end_x, end_y), 
                                color=color, thickness=2, tipLength=0.3)
            
            cv2.imwrite(debug_vdir_save_folder / img_path.name, img_vis_vdir)

            # draw last observed_zs
            img_vis_last_ob = img.copy()
            for cx, cy, w, h, id in debug_info["last_observed_zs"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_last_ob, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_last_ob, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_lastOb_save_folder / img_path.name, img_vis_last_ob)

            # draw prev center
            img_vis_prev_center = img.copy()
            for cx, cy, id in debug_info["previous_obs"]:
                cx = int(cx)
                cy = int(cy)
                id = int(id)
                color = get_color(id)
                cv2.circle(img_vis_prev_center, (int(cx), int(cy)), radius=5, color=color, thickness=-1)
                cv2.putText(img_vis_prev_center, label, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_prevCenter_save_folder / img_path.name, img_vis_prev_center)

            # draw first round assign
            img_vis_firstRound = img.copy()
            for cx, cy, w, h, id in debug_info["first_round_assign"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_firstRound, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_firstRound, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_firstAssign_save_folder / img_path.name, img_vis_firstRound)

            # draw second round assign
            img_vis_secondRound = img.copy()
            for cx, cy, w, h, id in debug_info["second_round_assign"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_secondRound, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_secondRound, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_secondAssign_save_folder / img_path.name, img_vis_secondRound)

            # draw third round assign
            img_vis_thirdRound = img.copy()
            for cx, cy, w, h, id in debug_info["third_round_assign"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_thirdRound, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_thirdRound, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_thirdAssign_save_folder / img_path.name, img_vis_thirdRound)

            # draw newly create
            img_vis_newlyCreate = img.copy()
            for cx, cy, w, h, id in debug_info["newly_created"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_newlyCreate, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_newlyCreate, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_newlyCreate_save_folder / img_path.name, img_vis_newlyCreate)

            # draw newly delete
            img_vis_newlyDelete = img.copy()
            for cx, cy, w, h, id in debug_info["newly_deleted"]:
                x1 = int(cx - 0.5 * w)
                y1 = int(cy - 0.5 * h)
                x2 = int(cx + 0.5 * w)
                y2 = int(cy + 0.5 * h)
                id = int(id)
                color = get_color(id)
                cv2.rectangle(img_vis_newlyDelete, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{id}"
                cv2.putText(img_vis_newlyDelete, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            cv2.imwrite(debug_newlyDelete_save_folder / img_path.name, img_vis_newlyDelete)
        
    print("done")

