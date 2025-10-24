import collections

import cv2
import numpy as np



class UnifiedCMC:
    def __init__(self, min_features=30, method='orb', process_img_shape=(640, 640), ui_mask="", debug_mode=False, **kwargs):
        """
        Unified Camera Motion Compensation supporting ORB and Optical Flow
        method: 'orb' or 'optflow'
        """
        self.method = method
        self.min_features = min_features
        self.prev_gray = None
        self.prev_desc = None
        self.prev_kp = None
        self.prev_pts = None  # For optical flow
        self.curr_affine_matrix = np.eye(3)
        self.cumu_affine_matrix = np.eye(3)
        self.process_img_shape = process_img_shape  # process image shape, (H, W), smaller for efficiency)

        self.ori_img_shape = None  # original image shape, (H, W), set at runtime
        self.resize_ratio_x = None  # process image W / original image W, calculate at runtime
        self.resize_ratio_y = None  # process image H / original image H, calculate at runtime

        self.orb_config = {"orb_num": 1000}

        # my configuration
        self.optflow_config = {"maxCorners": 50, 
                               "qualityLevel": 0.01, 
                               "minDistance": 7,
                               "blockSize": 3,
                               "winSize": (15, 15),
                               "maxLevel": 3
                               }
        
        # UI mask(for debug video recording samples)
        self.ui_mask = None
        if len(ui_mask):  # load image
            try:
                ui_mask_img = cv2.imread(ui_mask, cv2.IMREAD_GRAYSCALE)
                ui_mask_img = cv2.resize(ui_mask_img, (self.process_img_shape[1], self.process_img_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                self.ui_mask = (ui_mask_img > 127).astype(np.uint8)  
            except:
                print(f"failed to load ui mask:{ui_mask}")

        # for debug
        self.debug_mode = debug_mode  # for visualization 
        self.vis_img = None  # visualization image
        self.feature_match_num = 0  # number of matching feature points 
        self.prev_pts_num = 0


    def _get_mask_by_detection(self, img, dets):
        if self.ui_mask is None:
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            if self.ui_mask.shape[0] != img.shape[0] or self.ui_mask.shape[1] != img.shape[1]:  # resize
                mask = cv2.resize(self.ui_mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
            else:
                mask = self.ui_mask
        
        for cx, cy, w, h in dets[:, :4]:
            x1 = max(int(cx - 0.5 * w), 0)
            y1 = max(int(cy - 0.5 * h), 0)
            x2 = min(int(cx + 0.5 * w), img.shape[1])
            y2 = min(int(cy + 0.5 * h), img.shape[0])
            mask[y1: y2, x1: x2] = 0
        return mask

    def _extract_features(self, img, mask):
        orb = cv2.ORB_create(self.orb_config["orb_num"])
        keypoints, descriptors = orb.detectAndCompute(img, mask)
        return keypoints, descriptors

    def _match_features(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def _estimate_rigid_transform(self, kp1, kp2, matches):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # encounter resize effect
        src_pts[:, 0] = src_pts[:, 0] / self.resize_ratio_x
        src_pts[:, 1] = src_pts[:, 1] / self.resize_ratio_y
        dst_pts[:, 0] = dst_pts[:, 0] / self.resize_ratio_x
        dst_pts[:, 1] = dst_pts[:, 1] / self.resize_ratio_y

        h, w = self.img_shape
        center = np.array([w / 2, h / 2])
        src_pts -= center
        dst_pts -= center
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        return matrix

    def _get_sparse_points(self, gray, mask):
        return cv2.goodFeaturesToTrack(gray, 
                                       maxCorners=self.optflow_config["maxCorners"], 
                                       qualityLevel=self.optflow_config["qualityLevel"], 
                                       minDistance=self.optflow_config["minDistance"], 
                                       mask=mask,
                                       blockSize=self.optflow_config["blockSize"])

    def _estimate_rigid_transform_from_points(self, pts1, pts2):
        h, w = self.ori_img_shape
        center = np.array([[w / 2, h / 2]])

        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        # undo resize effect
        pts1[:, 0] = pts1[:, 0] / self.resize_ratio_x
        pts1[:, 1] = pts1[:, 1] / self.resize_ratio_y
        pts2[:, 0] = pts2[:, 0] / self.resize_ratio_x
        pts2[:, 1] = pts2[:, 1] / self.resize_ratio_y
        # take center as origin
        pts1 = pts1 - center
        pts2 = pts2 - center
        matrix, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
        return matrix

    def update(self, curr_frame, dets):
        """
        to get cumulative affine matrix(first frame -> current frame, orginal scale)
        Args:
            curr_frame(numpy.ndarray): original image;
            dets(numpy.ndarray): yolo detections(original scale) for foreground, 
                shape=(#dets, 5), cx, cy, w, h, score;
        """
        self.ori_img_shape = curr_frame.shape[:2]  # set at runtime

        if len(curr_frame.shape) == 3:  # colorful bgr
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        elif len(curr_frame.shape) == 2:  # gray already
            curr_gray = curr_frame
        else:
            raise ValueError("something wrong")
        
        # resize for efficiency
        curr_gray = cv2.resize(curr_gray, (self.process_img_shape[1], self.process_img_shape[0]), 
                               interpolation=cv2.INTER_LINEAR)

        # calculate resize_ratio at runtime
        self.resize_ratio_x = self.process_img_shape[1] / self.ori_img_shape[1]
        self.resize_ratio_y = self.process_img_shape[0] / self.ori_img_shape[0]

        if (len(dets)):  # resize detection boxes
            org_dets = dets
            dets = org_dets.copy()
            dets[:, 0] = org_dets[:, 0] * self.resize_ratio_x  # cx
            dets[:, 1] = org_dets[:, 1] * self.resize_ratio_y  # cy
            dets[:, 2] = org_dets[:, 2] * self.resize_ratio_x  # w
            dets[:, 3] = org_dets[:, 3] * self.resize_ratio_y  # h

        
        mask = self._get_mask_by_detection(curr_gray, dets)

        if self.prev_gray is None:   # first frame
            self.prev_gray = curr_gray
            if self.method == 'orb':
                self.prev_kp, self.prev_desc = self._extract_features(curr_gray, mask)
            else:
                self.prev_pts = self._get_sparse_points(curr_gray, mask)
            return np.eye(3)

        # use 2 different methods to get transform matrix from prev_frame to curr_frame
        if self.method == 'orb':
            A = self._update_orb(curr_gray, mask)
        else:
            A = self._update_optical_flow(curr_gray, mask)



        self.curr_affine_matrix = np.eye(3)
        self.curr_affine_matrix[:2] = A
        self.cumu_affine_matrix = self.curr_affine_matrix @ self.cumu_affine_matrix

        if self.debug_mode:
            self.vis_img = self.draw_camera_info(self.vis_img)

        return self.curr_affine_matrix

    def _update_orb(self, curr_gray, mask):
        curr_kp, curr_desc = self._extract_features(curr_gray, mask)
        if (self.prev_desc is None or curr_desc is None or
            len(self.prev_desc) < self.min_features or len(curr_desc) < self.min_features):
            A = np.eye(2, 3)
        else:
            matches = self._match_features(self.prev_desc, curr_desc)
            if len(matches) < self.min_features:
                A = np.eye(2, 3)
            else:
                A = self._estimate_rigid_transform(self.prev_kp, curr_kp, matches)
                if A is None:
                    A = np.eye(2, 3)
        self.prev_gray = curr_gray
        self.prev_kp = curr_kp
        self.prev_desc = curr_desc
        return A

    def _update_optical_flow(self, curr_gray, mask):
        if self.prev_pts is None or len(self.prev_pts) < self.min_features:
            
            if self.debug_mode:
                self.vis_img = self._vis_opt_flow(curr_gray, mask, self.prev_pts, None, None)
                if self.prev_pts is not None:
                    self.prev_pts_num = len(self.prev_pts)
                else:
                    self.prev_pts_num = 0
                self.feature_match_num = 0

            self.prev_pts = self._get_sparse_points(self.prev_gray, mask)
            self.prev_gray = curr_gray

            return np.eye(2, 3)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, 
                                                       self.optflow_config["winSize"],
                                                       self.optflow_config["maxLevel"])

        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < self.min_features:
            A = np.eye(2, 3)
        else:
            A = self._estimate_rigid_transform_from_points(good_prev, good_curr)
            if A is None:
                A = np.eye(2, 3)

        if self.debug_mode:  # for debug
            self.vis_img = self._vis_opt_flow(curr_gray, mask, self.prev_pts, curr_pts, status)
            self.feature_match_num = len(good_prev)
            self.prev_pts_num = len(self.prev_pts)

        # if good_curr is None or len(good_curr) < self.min_features:
        #     print("extract")
        #     self.prev_pts = self._get_sparse_points(curr_gray, mask)
        # else:
        #     print("reuse")
        #     self.prev_pts = good_curr.reshape(-1, 1, 2)

        self.prev_pts = self._get_sparse_points(curr_gray, mask)
        self.prev_gray = curr_gray
        return A
    
    def global_to_local(self, dets):
        if len(dets) == 0:
            return dets.copy()

        # Convert from [cx, cy, w, h] to [x1, y1], [x2, y2]
        cxcy = dets[:, :2]
        wh = dets[:, 2:4]


        homog = np.hstack([cxcy, np.ones((cxcy.shape[0], 1))])  # (#dets, 3)
        new_cxcy = np.dot(homog, self.cumu_affine_matrix[:2].T)  # (#dets, 2)

        scale_ratio_x = np.sqrt(self.cumu_affine_matrix[0, 0]**2 + self.cumu_affine_matrix[1, 0]**2)
        scale_ratio_y = np.sqrt(self.cumu_affine_matrix[0, 1]**2 + self.cumu_affine_matrix[1, 1]**2)
        scale_ratio = 0.5 * (scale_ratio_x + scale_ratio_y)
        new_wh = wh * scale_ratio

        # take top left ponit as origin
        h, w = self.ori_img_shape
        center = np.array([w / 2, h / 2])
        new_cxcy = new_cxcy + center

        new_dets = dets.copy()
        new_dets[:, 0:2] = new_cxcy
        new_dets[:, 2:4] = new_wh
        return new_dets
    
    def local_to_global(self, dets):
        # only transform center point
        # w and h then is scaled by scale ratio 
        if len(dets) == 0:
            return dets.copy()

        cxcy = dets[:, :2]  # (#dets, 2)

        # take center ponit as origin
        h, w = self.ori_img_shape
        center = np.array([w / 2, h / 2])
        cxcy = cxcy - center

        
        # transform center point
        homog = np.hstack([cxcy, np.ones((cxcy.shape[0], 1))])  # (#dets, 3)
        inv_matrix = cv2.invertAffineTransform(self.cumu_affine_matrix[:2])  # (2, 3)
        new_cxcy = np.dot(homog, inv_matrix.T)  # (#dets, 2)

        # rescale w and h
        scale_ratio_x = np.sqrt(inv_matrix[0, 0]**2 + inv_matrix[1, 0]**2)
        scale_ratio_y = np.sqrt(inv_matrix[0, 1]**2 + inv_matrix[1, 1]**2)
        scale_ratio = 0.5 * (scale_ratio_x + scale_ratio_y)
        new_wh = dets[:, 2:4] * scale_ratio  # (#dets, 2)

        new_dets = dets.copy()
        new_dets[:, 0:2] = new_cxcy
        new_dets[:, 2:4] = new_wh
        return new_dets
    
    def compensate_for_current_frame(self, dets):
        # only transform center point
        # w and h then is scaled by scale ratio 
        if len(dets) == 0:
            return dets.copy()
        
        cxcy = dets[:, :2]  # (#dets, 2)

        # take center ponit as origin
        h, w = self.ori_img_shape
        center = np.array([w / 2, h / 2])
        cxcy = cxcy - center
        
        # transform center point
        homog = np.hstack([cxcy, np.ones((cxcy.shape[0], 1))])  # (#dets, 3)
        trans_matrix = self.curr_affine_matrix[:2]  # (2, 3)
        new_cxcy = np.dot(homog, trans_matrix.T)  # (#dets, 2)

        # take top left point as origin
        new_cxcy = new_cxcy + center

        # rescale w and h
        scale_ratio_x = np.sqrt(trans_matrix[0, 0]**2 + trans_matrix[1, 0]**2)
        scale_ratio_y = np.sqrt(trans_matrix[0, 1]**2 + trans_matrix[1, 1]**2)
        scale_ratio = 0.5 * (scale_ratio_x + scale_ratio_y)
        new_wh = dets[:, 2:4] * scale_ratio  # (#dets, 2)

        new_dets = dets.copy()
        new_dets[:, 0:2] = new_cxcy
        new_dets[:, 2:4] = new_wh
        return new_dets

    
    def global_to_local_for_velocity(self, vs):
        # convert velocity direction from global to local for display 
        # vs = (#trks, 2), 2 means vx, vy
    
        M = self.cumu_affine_matrix[:2, :2]
        scale_x = np.linalg.norm(M[:, 0])
        scale_y = np.linalg.norm(M[:, 1])
        R = np.zeros((2, 2))
        R[:, 0] = M[:, 0] / scale_x
        R[:, 1] = M[:, 1] / scale_y

        local_vs = np.dot(vs, R.T)  # (#trks, 2)
        return local_vs
    
    def _decompose_affine(self, A):
        # A: 2x3 or 3x3 affine matrix
        if A.shape == (3, 3):
            A = A[:2, :]  # reduce to 2x3

        a, b, tx = A[0]
        c, d, ty = A[1]

        # Translation
        translation = (tx, ty)

        # Scale
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)

        # Rotation (in radians)
        rotation_rad = np.arctan2(c, a)
        rotation_deg = np.degrees(rotation_rad)

        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "rotation_rad": rotation_rad,
            "rotation_deg": rotation_deg,
            "translation_x": tx,
            "translation_y": ty
        }
    
    def get_camera_info(self):
        cumu_affine_info = self._decompose_affine(self.cumu_affine_matrix)
        curr_affine_info = self._decompose_affine(self.curr_affine_matrix)

        camera_info = {}
        camera_info["cumu_camera_zoom"] = (cumu_affine_info["scale_x"] + cumu_affine_info["scale_y"]) * 0.5
        camera_info["cumu_camera_translation"] = (-1 * cumu_affine_info["translation_x"], -1 * cumu_affine_info["translation_y"])
        camera_info["cumu_camera_rotation"] = -1 * cumu_affine_info["rotation_deg"]

        camera_info["curr_camera_zoom"] = (curr_affine_info["scale_x"] + curr_affine_info["scale_y"]) * 0.5
        camera_info["curr_camera_translation"] = (-1 * curr_affine_info["translation_x"], -1 * curr_affine_info["translation_y"])
        camera_info["curr_camera_rotation"] = -1 * curr_affine_info["rotation_deg"]

        return camera_info
    
    def draw_camera_info(self, image, color=(255, 255, 255),
                     bg_color=(0, 0, 0), alpha=0.5,
                     margin=10, padding=8):
        """
        Draw camera info in a semi-transparent background panel (top-right).

        Args:
            image: HxWxC BGR image (uint8).
            color: text color (BGR).
            bg_color: background panel color (BGR).
            alpha: opacity of the background [0..1].
            margin: distance from edges in pixels.
            padding: inner padding inside panel.

        Returns:
            image (with drawing).
        """
        camera_info = self.get_camera_info()

        H, W = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 18

        # --- build display lines ---
        lines = []
        for key, val in camera_info.items():
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                line = f"{key}: ({float(val[0]):.1f}, {float(val[1]):.1f})"
            elif isinstance(val, (float, int, np.floating, np.integer)):
                if "zoom" in key.lower():
                    line = f"{key}: x{float(val):.2f}"
                elif "rot" in key.lower():
                    line = f"{key}: {float(val):.1f} deg"
                else:
                    line = f"{key}: {float(val):.3f}"
            else:
                line = f"{key}: {val}"
            lines.append(line)

        if self.debug_mode:
            lines.append(f"number of match: {self.feature_match_num}")
            lines.append(f"number of previous points: {self.prev_pts_num}")

        if not lines:
            return image

        # --- measure text for panel size ---
        widths = [cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines]
        max_w = max(widths)
        total_h = len(lines) * line_height

        x1 = W - margin
        x0 = x1 - (max_w + 2 * padding)
        y0 = margin
        y1 = y0 + (total_h + 2 * padding)

        # clamp within image
        x0 = max(0, x0)
        y0 = max(0, y0)

        # --- draw semi-transparent background ---
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, thickness=-1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

        # --- draw text ---
        for i, (text, tw) in enumerate(zip(lines, widths)):
            baseline_y = y0 + padding + (i + 1) * line_height - (line_height - 12)
            text_x = x1 - padding - tw  # right align
            cv2.putText(image, text, (text_x, baseline_y), font, font_scale,
                        color, thickness, lineType=cv2.LINE_AA)

        return image
    
    def _vis_opt_flow(self, curr_gray, mask, prev_pts, curr_pts, status, use_ori_size=True):
        curr_gray_masked = curr_gray * mask
        vis_img = cv2.cvtColor(curr_gray_masked, cv2.COLOR_GRAY2BGR)
        if use_ori_size:
            vis_img = cv2.resize(vis_img, (self.ori_img_shape[1], self.ori_img_shape[0]))

        if prev_pts is not None and curr_pts is not None and status is not None:
            for (p0, p1, st) in zip(prev_pts, curr_pts, status):
                if st == 1:  # successfully tracked
                    x0, y0 = p0.ravel()
                    x1, y1 = p1.ravel()

                    if use_ori_size:
                        x0 = x0 / self.resize_ratio_x
                        y0 = y0 / self.resize_ratio_y
                        x1 = x1 / self.resize_ratio_x
                        y1 = y1 / self.resize_ratio_y

                    # old point (green circle)
                    cv2.circle(vis_img, (int(x0), int(y0)), 3, (0, 255, 0), -1)

                    # new point (red circle)
                    cv2.circle(vis_img, (int(x1), int(y1)), 3, (0, 0, 255), -1)

                    # line connecting them (blue)
                    cv2.line(vis_img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)
        else:  # only prev_pts is given
            for p0 in prev_pts:
                x0, y0 = p0.ravel()
                if use_ori_size:
                    x0 = x0 / self.resize_ratio_x
                    y0 = y0 / self.resize_ratio_y
                # old point (green circle)
                cv2.circle(vis_img, (int(x0), int(y0)), 3, (0, 255, 0), -1)
                   

        return vis_img
    

    





if __name__ == "__main__":
    from pathlib import Path
    import shutil

    from tqdm import tqdm
    from object_detect import YoloPredictor
    import matplotlib.pyplot as plt


    

    predictor = YoloPredictor(onnx_path="my_script/yolov8n_visdrone_add_people_merged_640_640_addIssueData.onnx",
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


    img_dir = Path("test_gimbal_data/test_frames/0004_W")
    img_paths = sorted(list(img_dir.glob("*.jpg")))
    img_num = len(img_paths)
    resize_ratio = 0.75
    for img_idx, img_path in tqdm(enumerate(img_paths), total=img_num):
        if img_idx < 700 or img_idx > 900:
            continue
        img_id = int(img_path.stem)
        if img_id == 225:
            print("debug")
        img = cv2.imread(str(img_path))
        if resize_ratio < 0.9 or resize_ratio > 1.1:
            img = cv2.resize(img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)

        dets = predictor.predict(img)  # for removing moving foregrounds
        curr_affine_matrix = cmc.update(img, dets)

        img_vis = cmc.vis_img
        if img_vis is not None:
            cv2.imwrite(vis_dir / img_path.name, img_vis)

        
        
            
            



    