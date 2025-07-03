import onnxruntime
import numpy as np
import cv2




class YoloPredictor():
    """good old YOLO Detector"""

    def __init__(self, onnx_path="my_script/yolov8n_visdrone_2.onnx",
                 conf_threshold = 0.25,
                 nms_threshold = 0.7,
                 min_w = 10,
                 min_h = 10,
                 **kwargs
                 ):
        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.min_w = min_w
        self.min_h = min_h
        # load ONNX model
        self.session = onnxruntime.InferenceSession(
            self.onnx_path, 
            providers=["CPUExecutionProvider"])
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        
        print(f"model input name: {self.input_name}")
        print(f"model output name: {self.output_name}")

    def predict(self, img):  # BGR
        H, W = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]

        # Inference
        output = self.session.run([self.output_name], {self.input_name: img_input})[0]
        output = np.squeeze(output)

        # Postprocess
        boxes = output[:4, :].T  # [8400, 4] in [cx, cy, w, h]
        cls_scores = output[4:, :].T  # [8400, num_classes]
        confidences = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Scale boxes from 640x640 to original image size
        scale_x, scale_y = W / 640.0, H / 640.0
        boxes_scaled = []
        for cx, cy, w, h in boxes:
            cx *= scale_x
            cy *= scale_y
            w *= scale_x
            h *= scale_y
            boxes_scaled.append([cx, cy, w, h])
        boxes_scaled = np.array(boxes_scaled)

        # Convert to [x1, y1, x2, y2] for NMS
        boxes_xyxy = []
        for cx, cy, w, h in boxes_scaled:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xyxy = np.array(boxes_xyxy).astype(np.int32)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )

        # Collect results in [cx, cy, w, h, score] (all in original image scale)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                cx, cy, w, h = boxes_scaled[i]
                score = confidences[i]
                if w > self.min_w and h > self.min_h:  # filter out extreme small objects
                    results.append([cx, cy, w, h, score])

        return np.array(results)
        

class ReID():
    """a REID model for calculating cosine similarity"""

    def __init__(self, onnx_path="my_script/fastreid_model.onnx", feat_dim=128, input_size=128):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, img):
        # preprocess
        resize_img = cv2.resize(img, (self.input_size, self.input_size))
        input_img = resize_img[:, :, ::-1]  # BGR to RGB
        input_img = input_img.astype(np.float32) 
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0)  # add batch dimension (1,3,H,W)

        # inference
        output = self.session.run([self.output_name], {self.input_name: input_img})[0]  # (1, feat_num)
        output = np.squeeze(output)
        
        return output


if __name__ == "__main__":
    # predictor = YoloPredictor()
    # img = cv2.imread("imgs/frame_yuv/00000169.jpg")
    # results = predictor.predict(img)
    # img_vis = img.copy()
    # for cx, cy, w, h, score in results:
    #     x1 = int(cx - 0.5 * w)
    #     y1 = int(cy - 0.5 * h)
    #     x2 = int(cx + 0.5 * w)
    #     y2 = int(cy + 0.5 * h)
    #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0,0,255), 2)
    #     label = f" {score:.2f}"
    #     cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    # cv2.imwrite("result.jpg", img_vis)


    embedder = ReID()
    img1 = cv2.imread("imgs/vis_id/2.jpg")
    feat1 = embedder.embed(img1)
    img2 = cv2.imread("imgs/vis_id/27.jpg")
    feat2 = embedder.embed(img2)
    img3 = cv2.imread("imgs/vis_id/609.jpg")
    feat3 = embedder.embed(img3)
    img4 = cv2.imread("imgs/vis_id/610.jpg")
    feat4 = embedder.embed(img4)

    cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    print(f"similarity score: {cos_sim:.2f}")

    cos_sim = np.dot(feat3, feat4) / (np.linalg.norm(feat3) * np.linalg.norm(feat4))
    print(f"similarity score: {cos_sim:.2f}")
    print("done")