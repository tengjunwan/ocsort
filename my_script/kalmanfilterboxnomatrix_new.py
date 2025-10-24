from collections import deque
import warnings

import numpy as np
from utils import exp_saturate_by_age


class KalmanFilterBoxTrackerNoMatrix():
    count = 0

    def __init__(self, cx, cy, w, h, delta_t, **kwargs):
        dim_x = 7  # (x, y, s, r, x', y', s')
        dim_z = 4  # (x, y, s, r)

        s = w * h
        r = w / (h + 1e-6)

        # stable observed shape(EMA)
        self.canonical_s = s
        self.canonical_r = r
        self.shape_diff_thresh = 0.2
        self.shape_update_coeff = 0.9
        self.is_occluded = True
    
        # state (mean)
        self.x = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)

        # state (covariance)
        self.P = np.eye(dim_x, dtype=np.float32)               # uncertainty covariance
        self.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.P *= 10.

        # process uncertainty(diagonal = [1,1,1,1,10**-2,10**-2,10**-4])
        self.R = np.eye(dim_x, dtype=np.float32)               
        self.R[-1, -1] *= 0.01
        self.R[4:, 4:] *= 0.01

        # measurement uncertainty(diagonal = [1,1,10,10])
        self.Q = np.eye(dim_z, dtype=np.float32)               
        self.Q[2:, 2:] *= 10.

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()             
        self.P_post = self.P.copy()
        
        # attributes for Observation-centric Re-Update(ORU)
        self.observed = True # i.e. it has disappeared
        self.last_observed_z = self.x[:4]  # (4, 1)
        self.last_observed_age = -1
        self.last_observed_x = self.x.copy() 
        self.last_observed_P = self.P.copy()
        
        self.age = 0  # number of frames track has existed after initialization
        self.consecutive_missed_frames = 0  # number of frames since last successful update
        self.consecutive_hits = 1  # # number of consecutive frames with successful updates
        self.last_z_buffer = deque(maxlen=delta_t)  # hold last "delta_t+1" observations, element is either shape (4,1) or None
        self.v_direction = np.array([0, 0], dtype=np.float32)  # velocity direction estimated by observations

        self.id = KalmanFilterBoxTrackerNoMatrix.count + 1  # start from 1
        KalmanFilterBoxTrackerNoMatrix.count += 1

        # appearance embedding
        self.feat = None  # a N-dim vector
        self.alpha_f = 0.95
        self.update_det_conf_score = 0.9
        self.update_det_score = 0.0
        self.update_simiarity = 1

    def predict(self, is_virtual=False):
        # update age 
        if not is_virtual:
            self.age = self.age + 1

        # area is forced to be non-negative
        min_area = 215  # 15*15 pixels
        if((self.x[6]+self.x[2]) <= min_area):
            self.x[6] *= 0.0

        R = self.R
        x = self.x
        P = self.P

        # Mean
        for i in range(3):  # x,y,s
            self.x_prior[i,0] = x[i,0] + x[4+i,0]
        for i in range(3, 7):  # r, vx, vy, vs
            self.x_prior[i,0] = x[i,0] 

        

        # Covariance
        for i in range(3):  # var_x, var_y, var_s
            self.P_prior[i,i] = P[i,i] + 2*P[i,4+i] + P[4+i,4+i] + R[i,i]  
        for i in range(3, 7):  # var_r, var_vx, var_vy, var_s
            self.P_prior[i,i] = P[i,i] + R[i,i]
        for i in range(0, 3):  # cov(x,vx), cov(y,vy), cov(s,vs),
            self.P_prior[0+i,4+i] = P[0+i,4+i] + P[4+i,4+i]
            self.P_prior[4+i,0+i] = self.P_prior[0+i,4+i]
       

        self.x = self.x_prior.copy()
        self.P = self.P_prior.copy()

        ret_state = self._cxcysr2cxcywh(self.x[:4].flatten())  

        return ret_state  # (4,) cxcywh

    def _oru(self, z):  # Observation-centric Re-Update
        age_gap = self.age - self.last_observed_age 

        # get x y w h and then gerenate virtual measurements by linear interpolation
        x1, y1, s1, r1 = self.last_observed_z.flatten()
        w1 = np.sqrt(s1 * r1)
        h1 = np.sqrt(s1 / (r1 + 1e-6))
        x2, y2, s2, r2 = z.flatten()
        w2 = np.sqrt(s2 * r2)
        h2 = np.sqrt(s2 / (r2 + 1e-6))
        dx = (x2-x1) / age_gap
        dy = (y2-y1) / age_gap 
        dw = (w2-w1) / age_gap 
        dh = (h2-h1) / age_gap

        # last observed state saved is actually prior state(result of prediction)
        # udpate -> predict -> update -> predict -> ... update -> predict
        # since we will do z updating normally in self.update() later on
        self.x = self.last_observed_x
        self.P = self.last_observed_P
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        for i in range(age_gap - 1):
            x = x1 + (i+1) * dx 
            y = y1 + (i+1) * dy 
            w = w1 + (i+1) * dw 
            h = h1 + (i+1) * dh

            virtual_z = np.array([x, y, w, h])
            self.update(virtual_z, is_virtual=True)
            self.predict(is_virtual=True)

    def get_previous_obs_for_v_direction(self):
        prev_z = None
        # try to find previous z in buffer
        if len(self.last_z_buffer) > 1:
            for i in range(len(self.last_z_buffer)):
                prev_z = self.last_z_buffer[i]
                if prev_z is not None:
                    break
        # still not found then resort to self.last_observed_z
        if prev_z is None and self.last_observed_z is not None:
            prev_z = self.last_observed_z  # x, y, s, r

        return prev_z
    
    def get_state_with_id(self):
        cx, cy, w, h = self._cxcysr2cxcywh(self.x[:4].flatten())
        id = self.id

        return np.array([cx, cy, w, h, id], dtype=np.float32)
    
    def get_state(self):
        cx, cy, w, h = self._cxcysr2cxcywh(self.x[:4].flatten())

        return np.array([cx, cy, w, h], dtype=np.float32)
    
    def set_state(self, cx, cy, w, h):
        cx, cy, s, r = self._cxcywh2cxcysr(np.array([cx, cy, w, h], dtype=np.float32))
        self.x[:4] = np.array([cx, cy, s, r]).reshape(4, 1)

    def get_velocity(self):
        vx = self.x.flatten()[4]
        vy = self.x.flatten()[5]
        return np.array([vx, vy], dtype=np.float32)
    
    def set_velocity(self, vx, vy):
        self.x[4:6] = np.array([vx, vy]).reshape(2, 1)
    
    def get_last_observed_z(self):
        ret = self.last_observed_z
        if ret is not None:
            ret = self._cxcysr2cxcywh(ret.flatten())
        return ret
    
    def set_last_observed_z(self, cx, cy, w, h):
        cxcysr = self._cxcywh2cxcysr(np.array([cx, cy, w, h]))  # shape=(4,)
        shape = self.last_observed_z.shape
        self.last_observed_z = cxcysr.reshape(shape)

    def get_last_z_buffer(self):
        # return deep copy of 'self.last_z_buffer'
        ret = deque(maxlen=self.last_z_buffer.maxlen)
        for ele in self.last_z_buffer:
            if ele is None:
                ret.append(ele)
            elif isinstance(ele, np.ndarray):
                ele = self._cxcysr2cxcywh(ele)  # xysr -> xywh
                ret.append(ele)
            else:
                raise ValueError(f"self.last_z_buffer has element of invalid type: {type(ele)}!")
        
        return ret
    
    def set_last_z_buffer(self, last_z_buffer):
        if not isinstance(last_z_buffer, deque) and not isinstance(last_z_buffer, list):
            warnings.warn(f"invalid type of 'last_z_buffer': {type(last_z_buffer)}!")
            return 

        if len(last_z_buffer) != len(self.last_z_buffer):
            warnings.warn(f"need 'len(last_z_buffer)' to be {len(self.last_z_buffer)} while it's {len(last_z_buffer)}!")
            return
        
        for i, ele in enumerate(last_z_buffer):
            if ele is None:
                self.last_z_buffer[i] = None
            elif isinstance(ele, np.ndarray):
                shape = self.last_z_buffer[i].shape
                ele = self._cxcywh2cxcysr(ele)  # xywh -> xysr
                self.last_z_buffer[i] = ele.reshape(shape)
            else:
                pass  # invalid element



    def _estimate_speed_direction(self, z):
        curr_z = z
        prev_z = self.get_previous_obs_for_v_direction()

        if prev_z is not None:
            # calculate velocity direction based on observation
            cx1, cy1 = prev_z.flatten()[:2]
            cx2, cy2 = curr_z.flatten()[:2]
            speed = np.array([cx2-cx1, cy2-cy1])
            norm = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2) + 1e-6
            self.v_direction = speed / norm
        else:
            # no valid previous z found, no speed direction
            self.v_direction = np.array([0, 0], dtype=np.float32)

    def update(self, z, is_virtual=False):
        """
        z : (dim_z, 1) or (dim_z, ): array_like or None, format "cxcywh"
            measurement for this update. 
        """
        if z is None:
            self.consecutive_missed_frames += 1 
            self.consecutive_hits = 0
            self.is_shape_deviated = True
            if self.observed:  # first time not observed
                # save last observation for oru
                self.last_observed_age = self.age - 1  # age +1 in predict phase
                self.last_observed_x = self.x  # prior x
                self.last_observed_P = self.P  # prior P
            self.observed = False
            self.x_post = self.x.copy()  # directly copy
            self.P_post = self.P.copy()

            self.last_z_buffer.append(z)

            ret_state = self._cxcysr2cxcywh(self.x[:4].flatten())  
            return ret_state  # (4, ), cxcywh
        
        # z is not none 
        z = self._cxcywh2cxcysr(z)
        z = z.reshape(4,1)

        # estimate speed velocity 
        if not is_virtual:
            self._estimate_speed_direction(z)

        if not self.observed and not is_virtual:  
            self._oru(z)

        if not is_virtual:
            self.observed = True
            self.consecutive_missed_frames = 0 
            self.consecutive_hits += 1
            self.last_observed_z = z
            self.last_z_buffer.append(z)  # update last z buffer

        # check if z deviates from canonical shape
        self.is_shape_deviated = False
        if not is_virtual:
            s = z[2, 0]
            r = z[3, 0]
            self._update_canonical_shape(s, r)
            self.is_shape_deviated = self._check_shape_deviation(s, r)

        # normal correct(update) phase
        Q = self.Q
        x = self.x
        P = self.P

        # Mean
        for i in range(4):  # x, y, s, r
            self.x_post[i,0] = P[i,i] / (P[i,i]+Q[i,i]+1e-6) * (z[i,0]-x[i,0]) + x[i,0]
        for i in range(0,3):  # vx,vy,vs
            self.x_post[i+4,0] = P[i,i+4] / (P[i,i]+Q[i,i]+1e-6) * (z[i,0]-x[i,0]) + x[i+4,0]

        # Covariance
        for i in range(4):  # var_x, var_y, var_s, var_r
            self.P_post[i,i] = Q[i,i]*P[i,i] / (Q[i,i]+P[i,i])
        for i in range(4,7):  # var_vx, var_vy, var_vs
            self.P_post[i,i] = P[i,i] - P[i-4,i]**2 / (P[i-4,i-4]+Q[i-4,i-4])
        for i in range(3):  # cov_x_vx, cov_y_vy, cov_s_vs
            self.P_post[i,i+4] = Q[i,i]*P[i,i+4] / (P[i,i]+Q[i,i]) 
            self.P_post[i+4,i] = self.P_post[i,i+4]

        # only update position(cx, cy) if shape is deviated(e.g. partial detection due to occlusion)
        if self.is_shape_deviated:
            for i in range(2, 7):  # s, r, vx, vy, vs
                self.x_post[i,0] = self.x[i,0]
                
        
        
        self.x = self.x_post.copy()
        self.P = self.P_post.copy()


        ret_state = self._cxcysr2cxcywh(self.x[:4].flatten())

        return ret_state  # (4,) cxcywh

    
    
    def _cxcywh2cxcysr(self, cxcywh):
        cx, cy, w, h = cxcywh.flatten()
        shape = cxcywh.shape
        s = w * h
        r = w / (h + 1e-6)
        cxcysr = np.array([cx, cy, s, r], dtype=np.float32).reshape(shape)
        return cxcysr
    
    def _cxcysr2cxcywh(self, cxcysr):
        cx, cy, s, r = cxcysr.flatten()
        shape = cxcysr.shape
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        cxcywh = np.array([cx, cy, w, h], dtype=np.float32).reshape(shape)
        return cxcywh
    
    def _check_shape_deviation(self, s, r):
        canonical_w = np.sqrt(self.canonical_s * self.canonical_r)
        canonical_h = self.canonical_s / (canonical_w + 1e-6)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        w_diff = abs(canonical_w - w) / (canonical_w + 1e-6)
        h_diff = abs(canonical_h - h) / (canonical_h + 1e-6)
        shape_diff = max(w_diff, h_diff)

        # if s_diff > self.s_diff_thresh or r_diff > self.r_diff_thresh:
        # return False
        if shape_diff > self.shape_diff_thresh:
            return True
        else:
            return False

    def _get_dynamic_shape_update_coeff(self):
        min_shape_update_coeff = 0.1
        max_shape_update_coeff = self.shape_update_coeff
        tau = 10.0  # saturate frame ≈ 30
        dynamic_shape_update_coeff = exp_saturate_by_age(self.age, min_shape_update_coeff, max_shape_update_coeff, tau)
        return dynamic_shape_update_coeff


        
    def _update_canonical_shape(self, s, r):
        # self.canonical_s = self.canonical_s * self.shape_update_coeff + s * (1 - self.shape_update_coeff)
        # self.canonical_r = self.canonical_r * self.shape_update_coeff + r * (1 - self.shape_update_coeff)
        dynamic_shape_update_coeff = self._get_dynamic_shape_update_coeff()
        self.canonical_s = self.canonical_s * dynamic_shape_update_coeff + s * (1 - dynamic_shape_update_coeff)
        self.canonical_r = self.canonical_r * dynamic_shape_update_coeff + r * (1 - dynamic_shape_update_coeff)

    def update_appearance(self, feat, det_score):
        # first update, no matter what det_score is, it's a useful start
        if self.feat is None:  
            self.feat = feat
            self.update_det_score = det_score
            self.update_simiarity = 1.0  # get cos similarity for debug
            return 
        
        self.update_simiarity = self._cal_cos_similarity(self.feat, feat)  # get cos similarity for debug
        
        
        if self.update_det_score < self.update_det_conf_score:
            if det_score > self.update_det_score:  # more confident than previous update while no one confident enough update
                self.feat = feat
                self.update_det_score = det_score
            else:
                pass
        else:
            if det_score > self.update_det_score:  # confident enough then use EMA to average all confident updates
                # confident enough
                dynamci_alpha = self.alpha_f + \
                    (1 - self.alpha_f) * (1 - (det_score - self.update_det_conf_score) / (1 - self.update_det_conf_score))
                dynamci_alpha = min(dynamci_alpha, 1)
                self.feat = dynamci_alpha * self.feat + (1 - dynamci_alpha) * feat
                self.update_det_score = det_score

    # def update_appearance(self, feat, det_score):
    #     if self.feat is None:  
    #         self.feat = feat
    #         self.update_det_score = det_score
    #         self.update_simiarity = 1.0  # get cos similarity for debug
    #         return 
        
    #     self.update_simiarity = self._cal_cos_similarity(self.feat, feat)  # get cos similarity for debug
        
        
    #     dynamci_alpha = self.alpha_f + \
    #         (1 - self.alpha_f) * (1 - (det_score - self.update_det_conf_score) / (1 - self.update_det_conf_score))
    #     dynamci_alpha = min(dynamci_alpha, 1)
    #     self.feat = dynamci_alpha * self.feat + (1 - dynamci_alpha) * feat
    #     self.update_det_score = det_score

    def get_appearance(self):
        return self.feat
    
    def _cal_cos_similarity(self, feat1, feat2):
        # Compute cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0  # Avoid division by zero
        
        cos_similarity = dot_product / (norm1 * norm2)
        return cos_similarity

    

if __name__ == "__main__":
    from pathlib import Path
    import shutil
    import cv2

    class RedDotDetector():
        """simple red dot detector for exp"""

        def __init__(self):
            self.lower_red1 = np.array([0, 50, 50])     # First range for red (hue around 0°)
            self.upper_red1 = np.array([10, 255, 255])
            self.lower_red2 = np.array([170, 50, 50])   # Second range for red (hue around 180°)
            self.upper_red2 = np.array([180, 255, 255])
            self.smallest_area = 5
            
        def detect(self, frame):
            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create masks for both red ranges and combine them
            mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Find contours from the mask
            contours, _ = cv2.findContours(mask, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around detected red dots
            detect = False
            for contour in contours:
                if cv2.contourArea(contour) > self.smallest_area:  
                    x, y, w, h = cv2.boundingRect(contour)
                    detect = True
                    break
            # filter out none square detection
            if detect:
                ratio = w / (h + 1e-6)
                if ratio > 1.5 or ratio < 0.666:
                    detect = False

            if detect:
                cx = x + 0.5 * w
                cy = y + 0.5 * h
                return np.array([cx, cy, w, h], dtype=np.float32)
            else:
                return None
    

    def draw_box(frame, cx, cy, w, h, color):
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


    save_folder = Path("vis_kf")
    for folder in [save_folder]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()

    detector = RedDotDetector()

    cap = cv2.VideoCapture("red_dot2.mp4")  # or 0 for webcam

    np.set_printoptions(suppress=True, precision=3, linewidth=150)

    tracker = None
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"frame count: {frame_count}")
        # if frame_count ==  198:
        #     print("special")
        vis_frame = frame.copy()
        

        predict = None
        if tracker is not None:
            p_cx, p_cy, p_w, p_h = tracker.predict()
            print("prior P:")
            print(tracker.P)
            draw_box(vis_frame, p_cx, p_cy, p_w, p_h, (255,0,0))  # blue
            
        
        det = detector.detect(frame)
        
        if det is not None:
            d_cx, d_cy, d_w, d_h = det
            draw_box(vis_frame, d_cx, d_cy, d_w, d_h, (0,255,0))  # green
            if tracker is None:
                tracker = KalmanFilterBoxTrackerNoMatrix(d_cx, d_cy, d_w, d_h, 3)  # init
                print("post P:")
                print(tracker.P)
            else:
                cx, cy, w, h = tracker.update(np.array([d_cx, d_cy, d_w, d_h], dtype=np.float32))
                draw_box(vis_frame, cx, cy, w, h, (0,0,255))  # red
        else:
            if tracker is not None:
                cx, cy, w, h = tracker.update(None)
                draw_box(vis_frame, cx, cy, w, h, (0,0,255))  # red
                print("post P:")
                print(tracker.P)

        
        # Process the frame
        cv2.imwrite(save_folder / f"{frame_count:07d}.jpg", vis_frame)


    cap.release()