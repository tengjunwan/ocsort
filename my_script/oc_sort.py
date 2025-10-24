from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# from kalmanfilterbox import KalmanFilterBoxTracker
# from kalmanfilterboxnomatrix import KalmanFilterBoxTrackerNoMatrix as KalmanFilterBoxTracker
from kalmanfilterboxnomatrix_new import KalmanFilterBoxTrackerNoMatrix as KalmanFilterBoxTracker
from utils import iou_batch, speed_direction_batch, appearance_batch, exp_saturate_by_age

USE_MY_HUNGARIAN = True
if USE_MY_HUNGARIAN:
    from hungarian_algorithm import hungarian_rect


def linear_assignment(cost_matrix):
    if USE_MY_HUNGARIAN:
        total_cost, matching = hungarian_rect(cost_matrix)
    else:
        x, y = linear_sum_assignment(cost_matrix)
        matching = np.array(list(zip(x, y)))
    return matching


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
    update_similarity: float
    missed_frames: int



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
        # 1st round association para
        first_round_buffer_ratio=0.2, 
        first_round_buffer_speed=5.0,
        first_round_match_iou_threshold=0.3, 
        first_round_match_v_dir_weight=0.2,
        first_round_match_v_dir_cos_threshold=-0.5, 
        first_round_match_app_threshold=0.5, 
        first_round_match_app_weight=0.2, 
        # 2nd round association para
        second_round_buffer_ratio=0.2, 
        second_round_buffer_speed=5.0,
        second_round_match_iou_threshold=0.3, 
        second_round_match_v_dir_weight=0.2,
        second_round_match_v_dir_cos_threshold=-0.5,
        second_round_match_app_threshold=0.0, 
        second_round_match_app_weight=0.0, 
        # 3rd round association para
        third_round_buffer_ratio=1.0, 
        third_round_buffer_speed=80.0,
        third_round_match_iou_threshold=0.01,
        third_round_match_v_dir_weight=0.2,
        third_round_match_v_dir_cos_threshold=0.0,
        third_round_match_app_threshold=0.9, 
        third_round_match_app_weight=1.0, 

        delta_t=3, use_byte=True, low_det_thresh=0.2, max_track_num=40, 
        # create new track 
        create_new_track_det_thresh=0.65,
        create_new_track_det_thresh_stricter=0.8,
        create_new_track_iou_thresh_stricter=0.3,
        feat_dim=128, verbose=False, 
        **kwargs):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits

        # first round priority(with high confident detection): location > appearance 
        self.first_round_buffer_ratio = first_round_buffer_ratio
        self.first_round_buffer_speed = first_round_buffer_speed
        self.first_round_match_iou_threshold = first_round_match_iou_threshold
        self.first_round_match_v_dir_weight = first_round_match_v_dir_weight
        self.first_round_match_v_dir_cos_threshold = first_round_match_v_dir_cos_threshold
        self.first_round_match_app_threshold = first_round_match_app_threshold
        self.first_round_match_app_weight = first_round_match_app_weight

        # second round priority(with low confident detection): location > appearance 
        self.second_round_buffer_ratio = second_round_buffer_ratio
        self.second_round_match_iou_threshold = second_round_match_iou_threshold
        self.second_round_match_v_dir_weight = second_round_match_v_dir_weight
        self.second_round_match_v_dir_cos_threshold = second_round_match_v_dir_cos_threshold
        self.second_round_match_app_threshold = second_round_match_app_threshold
        self.second_round_match_app_weight = second_round_match_app_weight
        self.second_round_buffer_speed = second_round_buffer_speed
        
        # third round priority(with high confident detection): appearance > location
        self.third_round_buffer_ratio = third_round_buffer_ratio
        self.third_round_buffer_speed = third_round_buffer_speed
        self.third_round_match_iou_threshold = third_round_match_iou_threshold
        self.third_round_match_v_dir_weight = third_round_match_v_dir_weight
        self.third_round_match_v_dir_cos_threshold = third_round_match_v_dir_cos_threshold
        self.third_round_match_app_threshold = third_round_match_app_threshold
        self.third_round_match_app_weight = third_round_match_app_weight

        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = iou_batch
        self.use_byte = use_byte
        self.low_det_thresh = low_det_thresh
        self.max_track_num = max_track_num
        self.feat_dim = feat_dim
        self.verbose = verbose

        # create new track
        self.create_new_track_det_thresh = create_new_track_det_thresh
        self.create_new_track_det_thresh_stricter = create_new_track_det_thresh_stricter
        self.create_new_track_iou_thresh_stricter = create_new_track_iou_thresh_stricter
        KalmanFilterBoxTracker.count = 0

        # target awareness: if target tracklet is given, we prioritize tracking target instead of all tracklets 
        self.target_id = None
        self.target_exist = False
        self.target_consecutive_missed_frames = 0
        self.target_location = np.zeros(4, dtype=np.float32)  # cx, cy, w, h

    def update(self, yolo_dets, det_feats, det_feats_mask, projector=None, debug_mode=False):
        """
        Args:
            yolo_dets (numpy.ndarray): 
                shape (#det, 5), 5 means cx, cy, w, h, score; 
            det_feats (numpy.ndarray):
                shape (#det, feat_dim), appearance embeddings of detections;
            det_feats_mask (numpy.ndarray, bool):
                shape (#det, feat_dim), indicate vailidation appearance embeddings of detections;
            projector: 'None' or 'CoordinateProjector', used to project pixel coordinate to world coordinate;

        return:
            yolo_dets_with_id (numpy.ndarray):
                shape (#det, 6), 6 means cx, cy, w, h, score, id;
        """
        if debug_mode:
            debug_info = {
                "first_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "second_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "third_round_assign": [],  # (#num, 5), cx, cy, w, h, id
                "newly_created": [], # (#num, 5), cx, cy, w, h, id
                "newly_deleted": [], # (#num, 5), cx, cy, w, h, id
            }

        self.frame_count += 1


        if yolo_dets is None or len(yolo_dets) == 0:  # (#det, 5)
            yolo_dets = yolo_dets.reshape(-1, 5)  # (0, 5)

        if yolo_dets.shape[1] ==6:  # cx, cy, w, h, score, cls_id
            yolo_dets = yolo_dets[:, :5]
            
        # filter detections by score
        scores = yolo_dets[:, 4]  # (#det)
        high_conf_mask = scores > self.det_thresh
        mid_conf_mask = (scores > self.low_det_thresh) & ~high_conf_mask  # (0.1 < score <= det_thresh)

        dets_high = yolo_dets[high_conf_mask]      # (#dets, 5), Used for primary matching
        dets_second = yolo_dets[mid_conf_mask]     # (#dets, 5), Used for BYTE-style secondary matching
        det_feats_high = det_feats[high_conf_mask]
        det_feats_second = det_feats[mid_conf_mask]
        det_feats_mask_high = det_feats_mask[high_conf_mask]
        det_feats_mask_second = det_feats_mask[mid_conf_mask]

        # get attributes of the exsiting trackers for association
        trks = np.zeros((len(self.trackers), 5), dtype=np.float32)  # prediction locations, (#trks, 5), cx, cy, w, h, id
        trk_feats = np.zeros((len(self.trackers), self.feat_dim), dtype=np.float32)  # appearnces, (#trks, feat_dim), unnormalized
        v_directions = np.zeros((len(self.trackers), 2), dtype=np.float32)  # velocity directions, (#trks, 2), vx, vy, normalized or (0,0)
        previous_obs = np.zeros((len(self.trackers), 2), dtype=np.float32)  # previous observations, (#trks, 2), cx, cy
        consecutive_missed_frames = np.zeros(len(self.trackers), dtype=np.int32)  # for dynamic buffer ratio, (#trks,)
        self.target_exist = False
        for i in range(len(self.trackers)):
            # prediction locations
            cx, cy, w, h = self.trackers[i].predict()
            id = self.trackers[i].id
            trks[i] = np.array([cx, cy, w, h, id], dtype=np.float32)

            # target related
            if id == self.target_id:
                self.target_exist = True
                self.target_consecutive_missed_frames = self.trackers[i].consecutive_missed_frames
                self.target_location = np.array([cx, cy, w, h], dtype=np.float32)


            # appearnce
            app = self.trackers[i].get_appearance()
            if app is None:  # not seen yet due to ReID strategy
                app = np.zeros(self.feat_dim, dtype=np.float32)
            trk_feats[i] = app

            # velocity directions
            v_directions[i] = self.trackers[i].v_direction  # (vx, vy)

            # previous observations
            prev_z = self.trackers[i].get_previous_obs_for_v_direction()  # (cx, cy, s, r)
            previous_obs[i] = prev_z.flatten()[:2]  # (cx, cy)

            # consecutive_missed_frames
            consecutive_missed_frames[i] = self.trackers[i].consecutive_missed_frames

        # project from world coordinate to pixel coordinate
        if projector is not None:
            trks = projector.project_from_world_to_pixel(trks)

    
        # 1st round association(trackers vs high score detections)
        buffer_ratio = exp_saturate_by_age(
            consecutive_missed_frames,  
            vmin=0.0, 
            vmax=self.first_round_buffer_ratio, 
            tau=self.first_round_buffer_speed)
        matched, unmatched_dets, unmatched_trks = self.oc_associate(
            dets_high, trks, self.first_round_match_iou_threshold, buffer_ratio,
            v_directions, previous_obs, self.first_round_match_v_dir_weight, self.first_round_match_v_dir_cos_threshold,
            det_feats_high, trk_feats, self.first_round_match_app_weight, self.first_round_match_app_threshold)

        if self.verbose:
            print(f"======first round match({self.frame_count}-th frame)======")
            print(f"matched num: {len(matched)}")

        for det_idx, trk_idx in matched:   # update trackers by matched detection immediately
            if projector is not None:
                self.trackers[trk_idx].update(projector.project_from_pixel_to_world(dets_high[det_idx, :4]))
            else:
                self.trackers[trk_idx].update(dets_high[det_idx, :4])  # update location
            if det_feats_mask_high[det_idx]:
                self.trackers[trk_idx].update_appearance(det_feats_high[det_idx], dets_high[det_idx, 4])  # update appearance

            if debug_mode:
                cx, cy, w, h = dets_high[det_idx, :4]
                id = self.trackers[trk_idx].id
                debug_info["first_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))

        # 2nd round association, Bytetrack(left trackers + low score detections)
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            left_trks = trks[unmatched_trks]  # iou
            left_v_directions = v_directions[unmatched_trks]  # v direction
            left_previous_obs = previous_obs[unmatched_trks]  # v direction
            left_trk_feats = trk_feats[unmatched_trks]  # appearnce

            # dynamic buffer ratio
            left_consecutive_missed_frames = consecutive_missed_frames[unmatched_trks]
            left_buffer_ratios = exp_saturate_by_age(
                left_consecutive_missed_frames,  
                vmin=0.0, 
                vmax=self.second_round_buffer_ratio, 
                tau=self.second_round_buffer_speed)
            
            # association
            matched_in_left, unmatched_dets_in_sec, unmatched_trks_in_left = self.oc_associate(
                    dets_second, left_trks, self.second_round_match_iou_threshold, left_buffer_ratios,
                    left_v_directions, left_previous_obs, self.second_round_match_v_dir_weight, self.second_round_match_v_dir_cos_threshold,
                    det_feats_second, left_trk_feats, self.second_round_match_app_weight, self.second_round_match_app_threshold
                    )
            
            # update trackers by matched detection immediately
            for det_idx_in_sec, trk_idx_in_left in matched_in_left:
                trk_idx = unmatched_trks[trk_idx_in_left]
                if projector is not None:  # update location
                    self.trackers[trk_idx].update(projector.project_from_pixel_to_world(dets_second[det_idx_in_sec, :4]))  
                else:
                    self.trackers[trk_idx].update(dets_second[det_idx_in_sec, :4])  
                if det_feats_mask_second[det_idx_in_sec]:
                    self.trackers[trk_idx].update_appearance(det_feats_second[det_idx_in_sec],  # update appereance
                                                            dets_second[det_idx_in_sec, 4])

                if debug_mode:
                    cx, cy, w, h = dets_second[det_idx_in_sec, :4]  
                    id = self.trackers[trk_idx].id
                    debug_info["second_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))
            
            # update unmatched trackers 
            unmatched_trks = unmatched_trks[unmatched_trks_in_left] if len(unmatched_trks_in_left) else np.array([], dtype=np.int32)

            if self.verbose:
                print(f"======second round match({self.frame_count}-th frame)======")
                print(f"matched num: {len(matched_in_left)}")
                

        # 3rd round association(left trackers + left high score detections): buffered IoU + velocity direction + appearance
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_trks = trks[unmatched_trks]                  # BIoU
            left_dets_high = dets_high[unmatched_dets]        # BIoU + v direction
            left_v_directions = v_directions[unmatched_trks]  # v direction
            left_previous_obs = previous_obs[unmatched_trks]  # v direction
            left_det_feats_high = det_feats_high[unmatched_dets]  # appearance
            left_trk_feats = trk_feats[unmatched_trks]  # appearnce
            left_consecutive_missed_frames = consecutive_missed_frames[unmatched_trks]

            # choose dynamic buffer ratios according to missed frames
            left_buffer_ratios = exp_saturate_by_age(
                left_consecutive_missed_frames,  
                vmin=0.2, 
                vmax=self.third_round_buffer_ratio, 
                tau=self.third_round_buffer_speed)


            matched_in_left, unmatched_dets_in_left, unmatched_trks_in_left = self.oc_associate(
                    left_dets_high, left_trks, self.third_round_match_iou_threshold, left_buffer_ratios,
                    left_v_directions, left_previous_obs, self.third_round_match_v_dir_weight, self.third_round_match_v_dir_cos_threshold,
                    left_det_feats_high, left_trk_feats, self.third_round_match_app_weight, self.third_round_match_app_threshold
                    )
            
            # update trackers by matched detection immediately
            for det_idx_in_left, trk_idx_in_left in matched_in_left:   
                det_idx = unmatched_dets[det_idx_in_left]
                trk_idx = unmatched_trks[trk_idx_in_left]
                if projector is not None:  # update location
                    self.trackers[trk_idx].update(projector.project_from_pixel_to_world(dets_high[det_idx, :4]))  
                else:
                    self.trackers[trk_idx].update(dets_high[det_idx, :4])  

                if det_feats_mask_high[det_idx]:
                    self.trackers[trk_idx].update_appearance(det_feats_high[det_idx], dets_high[det_idx, 4])  # update appearance
                if debug_mode:
                    cx, cy, w, h = dets_high[det_idx, :4]
                    id = self.trackers[trk_idx].id
                    debug_info["third_round_assign"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))

            # update unmatched trackers and unmatched detections
            unmatched_dets = unmatched_dets[unmatched_dets_in_left] if len(unmatched_dets_in_left) else np.array([], dtype=np.int32)
            unmatched_trks = unmatched_trks[unmatched_trks_in_left] if len(unmatched_trks_in_left) else np.array([], dtype=np.int32)

            if self.verbose:
                print(f"======third round match({self.frame_count}-th frame)======")
                print(f"matched num: {len(matched_in_left)}")
            
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
            cx, cy, w, h, score = dets_high[det_idx]
            if len(self.trackers) > self.max_track_num:  # prevent too many trackers
                continue
            
            # conditions when we don't create new tracklet
            near_target = False
            if self.target_exist:
                # check if the detection is near target
                buffer_ratio = exp_saturate_by_age(
                    self.target_consecutive_missed_frames,  
                    vmin=0.2, 
                    vmax=self.third_round_buffer_ratio, 
                    tau=self.third_round_buffer_speed
                    )
                iou_with_target = iou_batch(np.array([[cx, cy, w, h]]), self.target_location.reshape(-1, 4), 0.0, buffer_ratio)[0, 0]
                near_target = iou_with_target > 0.0001 
            
            if near_target:  # when near target, we create new tracklet by stricter standard
                if score < self.create_new_track_det_thresh_stricter:
                    continue
                if not det_feats_mask_high[det_idx]:  # no appearance embedding 
                    continue
                    
            else:  # when far away from target, we create new track if score of detection is high enough
                if score < self.create_new_track_det_thresh:
                    continue

            if len(trks) != 0:
                # do not create track by when detection overlaps
                iou_with_trks = iou_batch(np.array([[cx, cy, w, h]]), trks[:, :4])  
                if np.max(iou_with_trks) > self.create_new_track_iou_thresh_stricter:
                    continue

            # create new tracklet
            if projector is not None:
                cx, cy, w, h = projector.project_from_pixel_to_world(np.array([cx, cy, w, h]))
            new_tracker = KalmanFilterBoxTracker(cx, cy, w, h, self.delta_t)
            if det_feats_mask_high[det_idx]:
                new_tracker.update_appearance(det_feats_high[det_idx], dets_high[det_idx, 4])
            self.trackers.append(new_tracker)
            if debug_mode:
                cx, cy, w, h = dets_high[det_idx, :4]
                id = self.trackers[-1].id
                debug_info["newly_created"].append(np.array([cx, cy, w, h, id], 
                                                   dtype=np.float32))
        
        if debug_mode:
            return self._return_trackers(), debug_info
        else:
            return self._return_trackers()
        
    def correct_gimbal_status(self,
                              projector, 
                              theta, 
                              phi, 
                              zoom,
                              height):
        
        if projector is None:
            return self._return_trackers()
        
        theta_before = projector.theta
        phi_before = projector.phi
        zoom_before = projector.zoom
        height_before = projector.height
        
        for i in range(len(self.trackers)):
            # =====change the gimbal status(old)=====
            projector.set_gimbal_status(theta=theta_before, phi=phi_before, zoom=zoom_before, height=height_before)

            # =====part 1: from world to pixel=====
            # ===state x===
            world_det = self.trackers[i].get_state()  # load 3d world detection, shape (4,) = (xw, zw, ww, hh)
            world_velocity = self.trackers[i].get_velocity()  # load 3d world velocity, shape (2,) = (vxw, vzw)
            pixel_det, pixel_velocity = projector.project_velocity_from_world_to_pixel(world_det, world_velocity)
            # ===last observed z===
            world_det_last_ob = self.trackers[i].get_last_observed_z()  # shape (4,) = (xw, zw, ww, hh)
            pixel_det_last_ob = projector.project_from_world_to_pixel(world_det_last_ob)
            # ===last observed buffer===
            last_z_buffer = self.trackers[i].get_last_z_buffer()  # deque
            for j, ele in enumerate(last_z_buffer):
                if ele is None:
                    continue
                elif isinstance(ele, np.ndarray):  # (4, 1), xywh
                    new_ele = projector.project_from_world_to_pixel(ele.flatten())  # (4, )
                    last_z_buffer[j] = new_ele.reshape(ele.shape)  # (4, 1)
                else:
                    pass

            # =====change the gimbal status(new/correct)=====
            projector.set_gimbal_status(theta=theta, phi=phi, zoom=zoom, height=height)

            # =====part 2: from pixel to world=====
            # ===state x===
            new_world_det, new_world_velocity = projector.project_velocity_from_pixel_to_world(pixel_det, pixel_velocity)
            self.trackers[i].set_state(new_world_det[0], new_world_det[1], new_world_det[2], new_world_det[3])
            self.trackers[i].set_velocity(new_world_velocity[0], new_world_velocity[1])
            # ===last observed z===
            new_world_det_last_ob = projector.project_from_pixel_to_world(pixel_det_last_ob)
            self.trackers[i].set_last_observed_z(new_world_det_last_ob[0], new_world_det_last_ob[1], 
                                                 new_world_det_last_ob[2], new_world_det_last_ob[3])
            # ===last observed buffer===
            for j, ele in enumerate(last_z_buffer):
                if ele is None:
                    continue
                elif isinstance(ele, np.ndarray):  # (4, 1), xywh
                    new_ele = projector.project_from_pixel_to_world(ele.flatten())
                    last_z_buffer[j] = new_ele.reshape(ele.shape)
                else:
                    pass
            self.trackers[i].set_last_z_buffer(last_z_buffer)


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
            update_similarity =trk_obj.update_simiarity
            missed_frames = trk_obj.consecutive_missed_frames
            rtn_tracker = Track(cx, cy, w, h, id, detected, vx, vy, is_occluded, 
                                canonical_s, canonical_r, update_similarity, missed_frames)
            rtn_trackers.append(rtn_tracker)
        return rtn_trackers
    
    def set_target(self, target_id):
        self.target_id = target_id
    
    @staticmethod
    def oc_associate(detections, trackers, iou_threshold, buffer_ratio,  # locations
                 velocities, previous_obs, v_dir_weight, v_dir_cos_threshold,  # velocity direction
                 det_feats, trk_feats, app_weight, app_threshold):     # appearance
        """
        Associates detections with trackers based on IoU and optional velocity difference cost.
    
        Args:
            detections (numpy.ndarray): Detection results from a detector (e.g., YOLO),
                shape = (num_detections, 5), where each detection is (cx, cy, w, h, score).
            trackers (numpy.ndarray): Predicted locations from a tracker (e.g., KalmanFilter),
                shape = (num_trackers, 5), where each tracker is (cx, cy, w, h, id).
            iou_threshold (float): Minimum IoU required to associate a detection with a tracker.
            buffer_ratio (numpy.ndarray or float): Buffer Prediction boxes of trackers.
            velocities (numpy.ndarray): Velocity vectors predicted by KalmanFilter,
                shape = (num_trackers, 2), where each vector is (vx, vy).
            previous_obs (numpy.ndarray): Previous observations used to compute velocity differences,
                shape = (num_trackers, 2), (center_x, center_y) per tracker.
            v_dir_weight (float): Weighting factor for velocity direction difference in the matching cost.
            v_dir_cos_threshold (float): minimal cosΔθ required to associate a detection with a tracker.
            det_feats (numpy.ndarray): Appereance embeddings of detections
                shape = (num_detections, feat_dim)
            trk_feats (numpy.ndarray): Appereance embeddings of trackers
                shape = (num_trackers, feat_dim)
            app_weight (float): base weight for Apperance rewarding matrix
            app_threshold (float): Minimum appearance similarity to associate a detection with a tracker.
    
        Returns:
            matches (numpy.ndarray): Matched detection-tracker pairs,
                shape = (num_matches, 2), each row is (detection_idx, tracker_idx).
            unmatched_detections (numpy.ndarray): Indices of unmatched detections,
                shape = (num_unmatched_detections,).
            unmatched_trackers (numpy.ndarray): Indices of unmatched trackers,
                shape = (num_unmatched_trackers,).
        """  
        if len(trackers)==0 or (len(detections)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.arange(len(trackers))

        # part 1: calculate iou reward matrix
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4], 0.0, buffer_ratio)  # (#dets, #trks), reward matrix
        non_overlap_mask = iou_matrix < 0.00001

        # part 2: calculate v_directioin reward matrix(#dets, #trks)
        X, Y = speed_direction_batch(detections, previous_obs)  # (#trks, #dets)
        inertia_X, inertia_Y = velocities[:,0], velocities[:,1]  # (#trks,)
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)  # (#trks, #dets)
        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)  # (#trks, #dets)
        diff_angle_cos = inertia_X * X + inertia_Y * Y # (#trks, #dets), cosθ
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle_cos = diff_angle_cos.T  # (#dets, #trks)
        diff_angle = np.arccos(diff_angle_cos)  # (#dets, #trks), θ ∈ [0, pi]
        diff_angle_reward = (np.pi /2.0 - np.abs(diff_angle)) / np.pi  # diff_angle_reward ∈ [-0.5, 0.5]
        scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)  # (#dets, #trks)
        weighted_angle_matrix = diff_angle_reward * v_dir_weight * scores  # (#dets, #trks)
        angle_not_ok_mask = diff_angle_cos < v_dir_cos_threshold  # (#dets, #trks)

        # part 3: calculate appearance reward matrix
        raw_app_matrix = appearance_batch(det_feats, trk_feats)  # (#dets, #trks)
        app_matrix = raw_app_matrix.copy()
        invaid_mask = non_overlap_mask | angle_not_ok_mask
        app_matrix[invaid_mask] = -1  # non-overlap and angle not ok = not similar at all

        # # adaptive weighting：find maximum value and 2nd maximum value of each row
        # if app_matrix.shape[1] >= 2:
        #     row_top2 = np.partition(app_matrix, -2, axis=1)[:, -2:]  # (#dets, 2)
        #     row_max = np.max(row_top2, axis=1)  # (#dets,)
        #     row_second_max = np.min(row_top2, axis=1)  # (#dets,)
        #     row_adaptive_weight = np.clip(row_max - row_second_max, 0, app_epsilon)  # (#dets,)
        # else:
        #     row_adaptive_weight = np.zeros(app_matrix.shape[0], dtype=np.float32)  # (#dets,)
        # # adaptive weighting：find maximum value and 2nd maximum value of each column
        # if app_matrix.shape[0] >= 2:
        #     col_top2 = np.partition(app_matrix, -2, axis=0)[-2:, :]  # (2, #trks)
        #     col_max = np.max(col_top2, axis=0)  # (#trks,)
        #     col_second_max = np.min(col_top2, axis=0)  # (#trks,)
        #     col_adaptive_weight = np.clip(col_max - col_second_max, 0, app_epsilon)  # (#trks,)
        # else:
        #     col_adaptive_weight = np.zeros(app_matrix.shape[1], dtype=np.float32)  # (#trks,)
        # app_adaptive_weight = (row_adaptive_weight[..., None] + col_adaptive_weight[None, ...]) * 0.5
        weighted_app_matrix = app_weight * app_matrix


        # form cost matrix 
        cost_matrix = -(iou_matrix + weighted_angle_matrix + weighted_app_matrix)
        matched_indices = linear_assignment(cost_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        # filter out matching (minimal IoU and appearance similarity requirement)
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold  # minimal IoU requirement
                or app_matrix[m[0], m[1]] < app_threshold  # minimal appearance requirement
                or diff_angle_cos[m[0], m[1]] < v_dir_cos_threshold  # minimal v direction(cos) requirement
                ):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


if __name__ == "__main__":
    pass
