from ultralytics import YOLO
import cv2
import numpy as np
import torch
from tqdm import tqdm

import os
import os.path as osp
from pathlib import Path
import shutil

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


class YOLOv8Predictor:
    
    def __init__(self, model_path="pretrained/yolov8n_visdrone_2.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device
        self.model.fuse()
        self.vehicle_ids = [0, 1, 2]  # car, bus, truck 
        self.imgsz = 640

    def inference(self, img, timer=None):
        img_info = {"id": 0}

        # Load image
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
            img = cv2.resize(img, (640, 640))
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = 1.0

        # Run detection
        if timer: timer.tic()
        results = list(self.model(img, imgsz=(self.imgsz, self.imgsz), verbose=True))
        boxes = results[0].boxes

        if boxes is None or boxes.xyxy.shape[0] == 0:
            return [None], img_info, [None]

        cls = boxes.cls.cpu().numpy()
        mask = np.isin(cls, self.vehicle_ids)
        if not np.any(mask):
            return [None], img_info, [None]

        xyxy = boxes.xyxy[mask].cpu().numpy()
        conf = boxes.conf[mask].cpu().numpy()
        
        output = np.concatenate([xyxy, conf[:, None]], axis=1)
        output_cls = cls[mask]

        return [torch.tensor(output, device=self.device)], img_info, output_cls
    

# 1. Create predictor
predictor = YOLOv8Predictor()  # or yolov8s.pt

# 2. Predict on images

img_dir = Path("imgs/uav0000306_00230_v")
img_paths= list(img_dir.glob("*.jpg"))
vis_dir = Path("vis_det")
if vis_dir.exists():
    shutil.rmtree(vis_dir)
vis_dir.mkdir(parents=True)

# img_paths = [Path("imgs/uav0000355_00001_v/0000001.jpg")]

for img_path in tqdm(img_paths, total=len(img_paths)):

    outputs, img_info, output_cls = predictor.inference(str(img_path))

    # 3. Draw and save result
    img = img_info["raw_img"]
    if outputs[0] is not None:
        for det, cls in zip(outputs[0].cpu().numpy(), output_cls):
            x1, y1, x2, y2, _ = det.astype(int)
            conf = det[4]
            if conf < 0.25:
                continue
            color = colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    save_path = vis_dir / img_path.name
    cv2.imwrite(str(save_path), img)
    # print("Saved to yolov8_vehicle_result.jpg")