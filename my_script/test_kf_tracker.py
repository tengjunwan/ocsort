from pathlib import Path
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from kalmanfilterboxnomatrix_new import KalmanFilterBoxTrackerNoMatrix as KalmanFilterBoxTracker


class RedAreaDetector():
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
        

img_dir = Path("imgs/red_box_moving2")
img_paths = sorted(img_dir.glob("*"))
img_num = len(img_paths)

vis_dir = Path("vis_kf")
if vis_dir.exists():
    shutil.rmtree(vis_dir)
vis_dir.mkdir(exist_ok=True)

detector = RedAreaDetector()
tracker = None
for img_path in tqdm(img_paths, total=img_num):
    img = cv2.imread(img_path)
    img_vis= img.copy()

    predict = None
    if tracker is not None:
        p_cx, p_cy, p_w, p_h = tracker.predict()
        draw_box(img_vis, p_cx, p_cy, p_w, p_h, (255,0,0))  # blue for prediction
            
        
    

    if tracker is not None:
        cx = tracker.x[0,0]
        cy = tracker.x[1,0]
        s = tracker.canonical_s
        r = tracker.canonical_r
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        shape_deviated = tracker.is_deviated
        if shape_deviated:
            draw_box(img_vis, cx, cy, w, h, (0,255,255))  # yellow for canonical box

    det = detector.detect(img)

    if det is not None:
        d_cx, d_cy, d_w, d_h = det
        draw_box(img_vis, d_cx, d_cy, d_w, d_h, (0,255,0))  # green for detection
        if tracker is None:
            tracker = KalmanFilterBoxTracker(d_cx, d_cy, d_w, d_h, 3)  # init
        else:
            cx, cy, w, h = tracker.update(np.array([d_cx, d_cy, d_w, d_h], dtype=np.float32))
            draw_box(img_vis, cx, cy, w, h, (0,0,255))  # red for correction
    else:
        if tracker is not None:
            cx, cy, w, h = tracker.update(None)
            draw_box(img_vis, cx, cy, w, h, (0,0,255))  # red for correction


    


        
    # Process the frame
    cv2.imwrite(vis_dir / img_path.name, img_vis)
