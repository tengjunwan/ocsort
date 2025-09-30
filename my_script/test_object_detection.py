from ultralytics import YOLO
import cv2
import numpy as np
import torch
from tqdm import tqdm

import os
import os.path as osp
from pathlib import Path
import shutil

from object_detect import YoloPredictor, OldYoloPredictor, YoloPredictorRect

# # Load model (you can use yolov8n.pt, yolov8s.pt, etc.)
# model = YOLO("pretrained/yolov8n.pt")  # or "yolov8s.pt", "your_model.pt"

# # Load an image
# img_path = "imgs/uav0000306_00230_v/0000001.jpg"
# img = cv2.imread(img_path)

# # Run detection
# results = model(img, imgsz=960)
# result = results[0]

# # Map class names to IDs
# names = result.names
# vehicle_ids = [k for k, v in names.items() if v in ["car", "bus", "truck"]]

# # Create mask for vehicle classes
# cls_array = result.boxes.cls.cpu().numpy()
# vehicle_mask = np.isin(cls_array, vehicle_ids)

# # Filter boxes
# result.boxes = result.boxes[vehicle_mask]

# # Show and save
# print(f"Detected {len(result.boxes)} vehicles (car/bus/truck)")
# # result.show()
# result.save(filename="vehicles_only.jpg")


colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 255, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 165, 255),
}


class RawYOLOv8Predictor:
    """
    official YOLO ultilization for ONNX comparison, support letterbox and direct rect inference
    """
    
    def __init__(self, model_path="pretrained/yolov8n_visdrone_2.pt", device="cuda", conf=0.25, iou=0.7):
        self.model = YOLO(model_path)
        self.device = device
        self.model.fuse()
        self.imgsz = 640
        self.conf = conf
        self.iou = iou

    def _sqaure_predict(self, img):
        H, W = img.shape[:2]
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz))


        # Run detection
        results = list(self.model(img_resized, verbose=False))
        boxes = results[0].boxes

        if boxes is None or boxes.xyxy.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        
        # mask = np.isin(cls, self.vehicle_ids)
        # if not np.any(mask):
        #     return [None], img_info, [None]

        # xyxy = boxes.xyxy[mask].cpu().numpy()
        # conf = boxes.conf[mask].cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        # rescale 
        scale_x, scale_y = W / 640.0, H / 640.0
        xyxy[:, 0] = scale_x * xyxy[:, 0]
        xyxy[:, 1] = scale_y * xyxy[:, 1]
        xyxy[:, 2] = scale_x * xyxy[:, 2]
        xyxy[:, 3] = scale_y * xyxy[:, 3]

        # xyxy to cxcywh
        cxcywh = np.zeros_like(xyxy, dtype=np.float32)
        cxcywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cxcywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        cxcywh[:, 2] = xyxy[:, 2] - xyxy[:, 0] 
        cxcywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        
        result = np.concatenate([cxcywh, conf[:, None], cls[:, None]], axis=1)

        return result  # (#dets, 6), xyxy, conf, cls
    
    def _default_predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        letterbox predict
        Returns ndarray (#dets, 6): [cx, cy, w, h, conf, cls] in ORIGINAL image coordinates.
        """
        H, W = img_bgr.shape[:2]

        r = self.model.predict(
            source=img_bgr,           # raw image; built-in letterbox is applied
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]

        if (r.boxes is None) or (len(r.boxes) == 0):
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = r.boxes.xyxy.cpu().numpy()     # already mapped to original W×H
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy()

        # xyxy -> cxcywh
        cxcywh = np.empty_like(xyxy, dtype=np.float32)
        cxcywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cxcywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0])
        cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1])

        out = np.concatenate([cxcywh, conf[:, None], cls[:, None]], axis=1).astype(np.float32)
        return out
    
    def predict(self, img: np.ndarray, square_pred=True) -> np.ndarray:
        if square_pred:
            return self._sqaure_predict(img)
        else:
            return self._default_predict(img)
    
    

def denoise_with_blur(img, method="gaussian", level=1):
    """
    Denoise noisy images by blurring.
    
    Args:
        img: input image (numpy array, BGR).
        method: "gaussian", "median", or "bilateral".
        level: integer >= 1, controls blur strength.
               higher = stronger smoothing.
    
    Returns:
        Blurred / denoised image.
    """
    level = max(1, int(level))

    if method == "gaussian":
        # kernel size must be odd
        k = 2 * level + 1
        return cv2.GaussianBlur(img, (k, k), sigmaX=level*0.8)

    elif method == "median":
        k = 2 * level + 1
        return cv2.medianBlur(img, k)

    elif method == "bilateral":
        # d: diameter of pixel neighborhood
        d = 5 + 2 * level
        sigmaColor = 20 * level
        sigmaSpace = 20 * level
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    else:
        return img
    
if __name__ == "__main__":
    # MODEL_PATH = Path("pretrained/yolov8n_visdrone_add_people.pt")
    # MODEL_PATH = Path("pretrained/yolov8n_visdrone_2.pt")
    # MODEL_PATH = Path("pretrained/yolov8n_visdrone_2.onnx")
    MODEL_PATH = Path("pretrained/yolov8n_visdrone_add_people_merged_640_640_addIssueData.onnx")
    model_format = MODEL_PATH.suffix

    IMG_DIR = Path("test_issues/low_detection_b_20250923_seg_1")
    # IMG_DIR = Path("test_issues/night_issue_seg_5")
    
    img_paths= list(IMG_DIR.glob("*.jpg"))
    VIS_DIR = Path("vis_det")
    if VIS_DIR.exists():
        shutil.rmtree(VIS_DIR)
    VIS_DIR.mkdir(parents=True)

    # CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "people"}
    CLASS_NAMES = {0: "vehicles", 1: "people"}
    PALETTE = [
        (37,255,225), (255,191,0), (0,255,0), (0,165,255), (255,0,255),
        (180,105,255), (0,215,255), (255,144,30), (144,238,144), (30,105,210)
    ]
    RESIZE_RATIO = 1.0

    # 1. Create predictor
    if model_format == ".pt":
        predictor = RawYOLOv8Predictor(model_path=MODEL_PATH, device="cuda")  
    elif model_format == ".onnx":
        # predictor = OldYoloPredictor(onnx_path=MODEL_PATH,
        #                              conf_threshold = 0.25,
        #                              nms_threshold = 0.7)
        predictor = YoloPredictor(onnx_path=MODEL_PATH,
                                  conf_threshold = 0.25,
                                  nms_threshold = 0.7)
        # predictor = YoloPredictorRect(onnx_path=MODEL_PATH,
        #                             conf_threshold = 0.25,
        #                             nms_threshold = 0.7)
    else:
        raise RuntimeError(f"no valid model: {MODEL_PATH}")


    
    # 2. Draw and save result
    for img_path in tqdm(img_paths, total=len(img_paths)):
        img = cv2.imread(str(img_path))
        if RESIZE_RATIO < 0.99:
            img = cv2.resize(img, None, fx=RESIZE_RATIO, fy=RESIZE_RATIO)
        # img = denoise_with_blur(img, "gaussian", level=6)
        vis_img = img.copy()
        # if model_format == ".pt":
        #     outputs, img_info, output_cls = predictor.predict(img)
            
        #     if outputs[0] is not None:
        #         for det, cls in zip(outputs[0].cpu().numpy(), output_cls):
        #             x1, y1, x2, y2, _ = det.astype(int)
        #             score = det[4]
        #             if score < 0.25:
        #                 continue
        #             cid = int(cls)
        #             color = PALETTE[cid]
        #             cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        #             label = f"{CLASS_NAMES[cid]}: {score:.2f}"
        #             cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # elif model_format == ".onnx":
        #     results = predictor.predict(img)
           
        #     for cx, cy, w, h, score, cid in results:
        #         x1 = int(cx - 0.5 * w)
        #         y1 = int(cy - 0.5 * h)
        #         x2 = int(cx + 0.5 * w)
        #         y2 = int(cy + 0.5 * h)
        #         cid = int(cid)
        #         color = PALETTE[cid % len(PALETTE)]
        #         cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        #         label = f"{CLASS_NAMES[cid]}: {score:.2f}"
        #         cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if model_format == ".pt":
            results = predictor.predict(img, square_pred=True)
        elif model_format == ".onnx":
            results = predictor.predict(img)
           
        for cx, cy, w, h, score, cid in results:
            x1 = int(cx - 0.5 * w)
            y1 = int(cy - 0.5 * h)
            x2 = int(cx + 0.5 * w)
            y2 = int(cy + 0.5 * h)
            cid = int(cid)
            color = PALETTE[cid % len(PALETTE)]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES[cid]}: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        save_path = VIS_DIR / img_path.name
        cv2.imwrite(str(save_path), vis_img)


        # print("Saved to yolov8_vehicle_result.jpg")