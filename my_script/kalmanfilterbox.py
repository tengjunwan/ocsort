from collections import deque

import numpy as np

class KalmanFilterBoxTracker():
    count = 0

    def __init__(self, cx, cy, w, h, delta_t):
        dim_x = 7  # (x, y, s, r, x', y', s')
        dim_z = 4  # (x, y, s, r)

        s = w * h
        r = w / (h + 1e-6)
    
        # state (mean)
        self.x = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)

        # state (covariance)
        self.P = np.eye(dim_x, dtype=np.float32)               # uncertainty covariance
        self.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.P *= 10.

        # state transition matrix
        self.A = np.array([[1, 0, 0, 0, 1, 0, 0], 
                           [0, 1, 0, 0, 0, 1, 0], 
                           [0, 0, 1, 0, 0, 0, 1], 
                           [0, 0, 0, 1, 0, 0, 0],  
                           [0, 0, 0, 0, 1, 0, 0], 
                           [0, 0, 0, 0, 0, 1, 0], 
                           [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32) 

        # measurement function              
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0], 
                           [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32) 

        # process uncertainty(diagonal = [1,1,1,1,10**-2,10**-2,10**-4])
        self.R = np.eye(dim_x, dtype=np.float32)               
        self.R[-1, -1] *= 0.01
        self.R[4:, 4:] *= 0.01

        # measurement uncertainty(diagonal = [1,1,10,10])
        self.Q = np.eye(dim_z, dtype=np.float32)               
        self.Q[2:, 2:] *= 10.

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x, dtype=np.float32)

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
        self.last_z_buffer = deque(maxlen=delta_t)  # hold last "delta_t+1" observations
        self.v_direction = np.array([0, 0], dtype=np.float32)  # velocity direction estimated by observations

        self.id = KalmanFilterBoxTracker.count + 1  # start from 1 
        KalmanFilterBoxTracker.count += 1

    def predict(self, is_virtual=False):
        # update age 
        if not is_virtual:
            self.age = self.age + 1

        A = self.A
        R = self.R

        # area is forced to be non-negative
        if((self.x[6]+self.x[2]) <= 0):
            self.x[6] *= 0.0

        # x_prior = Ax + Bu
        self.x = np.dot(A, self.x)

        # P_prior = APA' + R
        self.P = np.dot(np.dot(A, self.P), A.T) + R

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

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
        # udpate -> predict -> update -> .. update -> predict
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
            prev_z = self.last_observed_z

        return prev_z
    
    def get_last_observed_z(self):
        ret = self.last_observed_z
        if ret is not None:
            ret = self._cxcysr2cxcywh(ret.flatten())
        return ret

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
        z = z.reshape(4, 1)

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

        # normal correct(update) phase
        Q = self.Q
        C = self.C

        # y = z - Cx
        y = z - np.dot(C, self.x)

        # common subexpression for speed
        PCT = np.dot(self.P, C.T)

        # S = CPC' + Q
        S = np.dot(C, PCT) + Q
        SI = np.linalg.inv(S)

        # K = PC'inv(S)
        K = np.dot(PCT, SI)

        # x = x + Ky
        self.x = self.x + np.dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KC = self._I - np.dot(K, C)
        self.P = np.dot(I_KC, self.P)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        ret_state = self._cxcysr2cxcywh(self.x[:4].flatten())

        return ret_state  # (4,) cxcywh

    def get_state_with_id(self):
        cx, cy, w, h = self._cxcysr2cxcywh(self.x[:4].flatten())
        id = self.id

        return np.array([cx, cy, w, h, id], dtype=np.float32)
    
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
                tracker = KalmanFilterBoxTracker(d_cx, d_cy, d_w, d_h, 3)  # init
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
