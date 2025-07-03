import cv2
import numpy as np


class CMC():
    """CMC = camera motion compensation"""
    def __init__(self, min_features):
        # for feature matching(ORB) based method
        self.min_features = min_features
        self.prev_gray = None
        self.prev_desc = None
        self.prev_kp = None
        self.curr_affine_matrix = np.eye(3, 3)
        self.cumu_affine_matrix = np.eye(3, 3)
        self.img_shape = None

    def cal_2d_rigid_transformation(self, curr_frame, dets):
        """
        Compute the 2D affine transform from prev_frame to 
        curr_frame using ORB feature matching.
        """
        if self.img_shape is  not None:
            assert self.img_shape == curr_frame.shape[:2]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        mask = self._get_mask_by_detection(curr_frame, dets)
        
        
        # If no previous frame, just store current and return identity
        if self.prev_gray is None:
            self.img_shape = curr_gray.shape
            self.prev_gray = curr_gray
            self.prev_kp, self.prev_desc = self._extract_features(curr_gray, mask)
            self.curr_affine_matrix = np.eye(3, 3)
            return self.curr_affine_matrix

        curr_kp, curr_desc = self._extract_features(curr_gray, mask)

        # Handle insufficient features
        if (self.prev_desc is None or curr_desc is None or
            len(self.prev_desc) < self.min_features or len(curr_desc) < self.min_features):
            print("Not enough descriptors")
            A = np.eye(2, 3)
        else:
            matches = self._match_features(self.prev_desc, curr_desc)
            if len(matches) < self.min_features:
                print("Not enough good matches")
                A = np.eye(2, 3)
            else:
                A = self._estimate_rigid_transform(self.prev_kp, curr_kp, matches)
                if A is None:
                    A = np.eye(2, 3)

        # Save current as previous for next round
        self.prev_gray = curr_gray
        self.prev_kp = curr_kp
        self.prev_desc = curr_desc
        self.curr_affine_matrix = np.eye(3, 3)
        self.curr_affine_matrix[:2] = A  # use 3*3 homogenous matrix form 
        self.cumu_affine_matrix = self.curr_affine_matrix @ self.cumu_affine_matrix  # cumulative matrix form
        return self.curr_affine_matrix

    # def global_to_local(self, dets):
    #     if len(dets) == 0:
    #         return dets.copy()

    #     # Convert from [cx, cy, w, h] to [x1, y1], [x2, y2]
    #     cxcy = dets[:, :2]
    #     wh = dets[:, 2:4]
    #     x1y1 = cxcy - wh / 2
    #     x2y2 = cxcy + wh / 2

    #     # remove rotation
    #     clean_cumu_affine_matrix = self.cumu_affine_matrix.copy()
    #     clean_cumu_affine_matrix[0, 1] = 0
    #     clean_cumu_affine_matrix[1, 0] = 0

    #     corners = np.vstack([x1y1, x2y2])   # (2 * #dets, 2)
    #     homog = np.hstack([corners, np.ones((corners.shape[0], 1))])  # (2 * #dets, 3)
    #     new_corners = np.dot(homog, clean_cumu_affine_matrix[:2].T)  # (2 * #dets, 2)

    #     new_x1y1 = new_corners[:len(dets)]
    #     new_x2y2 = new_corners[len(dets):]

    #     new_cxcy = (new_x1y1 + new_x2y2) / 2
    #     new_wh = new_x2y2 - new_x1y1

    #     # take top left ponit as origin
    #     h, w = self.img_shape
    #     center = np.array([w / 2, h / 2])
    #     new_cxcy = new_cxcy + center

    #     new_dets = dets.copy()
    #     new_dets[:, 0:2] = new_cxcy
    #     new_dets[:, 2:4] = new_wh
    #     return new_dets
    
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
        h, w = self.img_shape
        center = np.array([w / 2, h / 2])
        new_cxcy = new_cxcy + center

        new_dets = dets.copy()
        new_dets[:, 0:2] = new_cxcy
        new_dets[:, 2:4] = new_wh
        return new_dets

    # def local_to_global(self, dets):
    #     if len(dets) == 0:
    #         return dets.copy()

    #     cxcy = dets[:, :2]  # (#dets, 2)

    #     # take center ponit as origin
    #     h, w = self.img_shape
    #     center = np.array([w / 2, h / 2])
    #     cxcy = cxcy - center

    #     wh = dets[:, 2:4]  # (#dets, 2)
    #     x1y1 = cxcy - wh / 2  # (#dets, 2)
    #     x2y2 = cxcy + wh / 2  # (#dets, 2)

    #     # remove rotation
    #     clean_cumu_affine_matrix = self.cumu_affine_matrix[:2]  # (2, 3)
    #     clean_cumu_affine_matrix[0, 1] = 0
    #     clean_cumu_affine_matrix[1, 0] = 0

    #     corners = np.vstack([x1y1, x2y2])  # (2 * #dets, 2)
    #     homog = np.hstack([corners, np.ones((corners.shape[0], 1))])  # (2 * #dets, 3)
    #     inv_matrix = cv2.invertAffineTransform(clean_cumu_affine_matrix)  # (2, 3)
    #     new_corners = np.dot(homog, inv_matrix.T)  # (2 * #dets, 2)

    #     new_x1y1 = new_corners[:len(dets)]
    #     new_x2y2 = new_corners[len(dets):]

    #     new_cxcy = (new_x1y1 + new_x2y2) / 2
    #     new_wh = new_x2y2 - new_x1y1

    #     new_dets = dets.copy()
    #     new_dets[:, 0:2] = new_cxcy
    #     new_dets[:, 2:4] = new_wh
    #     return new_dets
    
    def local_to_global(self, dets):
        # only transform center point
        # w and h then is scaled by scale ratio 
        if len(dets) == 0:
            return dets.copy()

        cxcy = dets[:, :2]  # (#dets, 2)

        # take center ponit as origin
        h, w = self.img_shape
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

    def _extract_features(self, img, mask):
        orb = cv2.ORB_create(1000)
        keypoints, descriptors = orb.detectAndCompute(img, mask)
        return keypoints, descriptors
    
    def _match_features(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # NORM_L2 for SIFT
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def _estimate_rigid_transform(self, kp1, kp2, matches):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        # take center ponit as origin
        h, w = self.img_shape
        center = np.array([w / 2, h / 2])
        src_pts = src_pts - center
        dst_pts = dst_pts - center
        # Use estimateAffinePartial2D for rigid transform (no scaling/shear)
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        return matrix
    
    def _get_mask_by_detection(self, img, dets):
        # shape of dets = (#dets, 5) cx, cy, w, h, score
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        for cx, cy, w, h, _ in dets:
            x1 = max(int(cx - 0.5 * w), 0)
            y1 = max(int(cy - 0.5 * h), 0)
            x2 = min(int(cx + 0.5 * w), img.shape[1])
            y2 = min(int(cy + 0.5 * h), img.shape[0])
            mask[y1: y2, x1: x2] = 0

        return mask
    
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
        affine_info = self._decompose_affine(self.cumu_affine_matrix)
        camera_info = {}
        camera_info["camera_zoom"] = (affine_info["scale_x"] + affine_info["scale_y"]) * 0.5
        camera_info["camera_translation"] = (-1 * affine_info["translation_x"], -1 * affine_info["translation_y"])
        camera_info["camera_rotation"] = -1 * affine_info["rotation_deg"]

        return camera_info
    
    def draw_camera_info(self, image, color=(0, 0, 255)):
        camera_info = self.get_camera_info()

        H, W = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 18

        # Convert info to display strings
        lines = [
            f"Zoom: x{camera_info['camera_zoom']:.2f}",
            f"Trans: ({camera_info['camera_translation'][0]:.1f}, {camera_info['camera_translation'][1]:.1f})",
            f"Rot: {camera_info['camera_rotation']:.1f} deg"
        ]

        # Draw lines from top-right corner downward
        for i, text in enumerate(lines):
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x = W - text_size[0] - 10  # Right-align
            y = 10 + i * line_height
            cv2.putText(image, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        return image  # Optional: return it if you want to chain ops

if __name__ == "__main__":
    from pathlib import Path
    import shutil

    from tqdm import tqdm
    from inference_onnx import YoloPredictor


    

    predictor = YoloPredictor()
    cmc = CMC(10)

    np.set_printoptions(precision=3, suppress=True)

    vis_dir = Path("vis_cmc")
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir()


    img_dir = Path("imgs/org_882be656fc94dade_1747721858000_seg_10")
    img_paths = sorted(list(img_dir.glob("*.jpg")))
    img_num = len(img_paths)
    for i, img_path in tqdm(enumerate(img_paths), total=img_num):
        img = cv2.imread(str(img_path))
        dets = predictor.predict(img)
        curr_affine_matrix = cmc.cal_2d_rigid_transformation(img, dets)

        global_dets = cmc.local_to_global(dets)
        local_dets = cmc.global_to_local(global_dets)

        img_vis = img.copy()
        # draw global cooridate 
        for i in range(len(dets)):
            cx, cy, w, h, score = local_dets[i]
            x1 = int(cx - 0.5 * w)
            y1 = int(cy - 0.5 * h)
            x2 = int(cx + 0.5 * w)
            y2 = int(cy + 0.5 * h)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), 
                            (0,0,255), 1)
            g_cx, g_cy, g_w, g_h, score = global_dets[i]
            label = f"({g_cx:.1f}, {g_cy:.1f}): {score:.2f}"
            cv2.putText(img_vis, label, (x2 - 80, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,0,255), 2)
            cmc.draw_camera_info(img_vis)
            
            cv2.imwrite(vis_dir / img_path.name, img_vis)
            
            



    