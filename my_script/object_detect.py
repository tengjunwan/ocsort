import onnxruntime
import numpy as np
import cv2




class OldYoloPredictor():  # obsolete
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

        
        # print(f"model input name: {self.input_name}")
        # print(f"model output name: {self.output_name}")

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
                cid = class_ids[i]
                if w > self.min_w and h > self.min_h:  # filter out extreme small objects
                    results.append([cx, cy, w, h, score, cid])

        return np.array(results)
    



class YoloPredictor:
    """YOLO ONNX inference with letterbox preproc and proper unletterbox postproc."""

    def __init__(self, onnx_path="my_script/yolov8n_visdrone_2.onnx",
                 conf_threshold=0.25,
                 nms_threshold=0.7,
                 min_w=10,
                 min_h=10,
                 input_size=640,
                 providers=("CPUExecutionProvider",)):
        self.onnx_path = onnx_path
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.input_size = int(input_size)

        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ----------------------------- #
    #           PREPROCESS          #
    # ----------------------------- #
    def _letterbox(self, img_bgr, new_shape=640, color=(114, 114, 114), stride=32):
        """
        Resize & pad to square new_shape while keeping aspect ratio.
        Returns:
            img_lb  : padded BGR image (new_shape x new_shape)
            ratio   : scale ratio used (w_ratio, h_ratio) – same value twice
            pad     : (dw, dh) padding applied on left/top
        """
        h0, w0 = img_bgr.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old) – same for both dims
        r = min(new_shape[0] / h0, new_shape[1] / w0)
        # New unpadded size
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        # Resize
        im = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Compute padding to reach target shape
        dw = new_shape[1] - new_unpad[0]  # width padding
        dh = new_shape[0] - new_unpad[1]  # height padding
        dw /= 2
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_lb = cv2.copyMakeBorder(im, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return img_lb, (r, r), (left, top)

    # ----------------------------- #
    #          POSTPROCESS          #
    # ----------------------------- #
    @staticmethod
    def _xywh2xyxy(x):
        # x: [N,4] in cx,cy,w,h
        y = np.empty_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    @staticmethod
    def _nms(boxes_xyxy, scores, iou_thr):
        """Simple class-agnostic NMS. boxes: [N,4], scores: [N]. Returns keep indices."""
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # IoU with the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int32)

    def predict(self, img_bgr):
        """
        Args:
            img_bgr: HxWx3 uint8 (BGR)
        Returns:
            np.ndarray of shape [M, 6]: [cx, cy, w, h, conf, cls] in ORIGINAL image coordinates (float32)
        """
        H, W = img_bgr.shape[:2]

        # ---- letterbox preprocess ----
        lb_img, (rx, ry), (padw, padh) = self._letterbox(img_bgr, new_shape=self.input_size)
        # to NCHW RGB float32 [0,1]
        blob = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,640,640]

        # ---- inference ----
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        out = np.squeeze(out)  # expect [C+4, N] or [N, C+4]

        # ---- parse output ----
        if out.ndim == 2 and out.shape[0] <= out.shape[1]:
            # shape [C+4, N] -> transpose to [N, C+4]
            out = out.T
        # out: [N, 4 + num_classes]
        boxes_cxcywh = out[:, :4]
        cls_scores = out[:, 4:]
        confidences = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        # ---- confidence filter ----
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # ---- map boxes from letterboxed space back to original image ----
        # 1) cxcywh -> xyxy in letterboxed coords
        boxes_xyxy = self._xywh2xyxy(boxes_cxcywh.copy())
        # 2) remove padding
        boxes_xyxy[:, [0, 2]] -= padw
        boxes_xyxy[:, [1, 3]] -= padh
        # 3) de-scale by ratio
        boxes_xyxy[:, [0, 2]] /= rx
        boxes_xyxy[:, [1, 3]] /= ry
        # 4) clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, W - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, H - 1)

        # ---- size filter (in original pixels) ----
        wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        size_mask = (wh[:, 0] >= self.min_w) & (wh[:, 1] >= self.min_h)
        if not np.any(size_mask):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = boxes_xyxy[size_mask]
        confidences = confidences[size_mask]
        class_ids = class_ids[size_mask]

        # ---- NMS (class-agnostic; switch to per-class if you prefer) ----
        keep = self._nms(boxes_xyxy, confidences, self.nms_threshold)
        if keep.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # ---- convert back to cx,cy,w,h in original coordinates ----
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
        w  = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        h  = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        res = np.stack([cx, cy, w, h, confidences, class_ids.astype(np.float32)], axis=1).astype(np.float32)
        return res
    



class YoloPredictorRect:
    """
    YOLO ONNX inference with **direct resize** to (H=480, W=640) and proper de-scale postproc.
    - No padding, no letterbox.
    - Works with Ultralytics-exported ONNX whose input is [1,3,480,640] (static) or dynamic but fed 480x640.
    """

    def __init__(self,
                 onnx_path: str,
                 conf_threshold: float = 0.25,
                 nms_threshold: float = 0.7,
                 min_w: int = 10,
                 min_h: int = 10,
                 input_h: int = 480,
                 input_w: int = 640,
                 providers=("CPUExecutionProvider",),
                 normalize: bool = True):
        self.onnx_path = onnx_path
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.normalize = bool(normalize)

        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Safety: check model input shape if static
        in_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,480,640] or [1,3,'height','width']
        if all(isinstance(x, int) for x in in_shape[-2:]):
            h_ok = (in_shape[-2] == self.input_h)
            w_ok = (in_shape[-1] == self.input_w)
            if not (h_ok and w_ok):
                raise ValueError(
                    f"Model expects HxW={in_shape[-2]}x{in_shape[-1]}, but wrapper configured for "
                    f"{self.input_h}x{self.input_w}. Adjust input_h/input_w or re-export the model."
                )

    # ----------------------------- #
    #          POSTPROCESS          #
    # ----------------------------- #
    @staticmethod
    def _xywh2xyxy(x):
        # x: [N,4] in cx,cy,w,h
        y = np.empty_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    @staticmethod
    def _nms(boxes_xyxy, scores, iou_thr):
        """Simple class-agnostic NMS. boxes: [N,4], scores: [N]. Returns keep indices."""
        if boxes_xyxy.size == 0:
            return np.empty((0,), dtype=np.int32)
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int32)

    def _preprocess(self, img_bgr):
        """
        Resize to (input_h,input_w). Return blob (1,3,H,W) float32 in [0,1] or unnormalized if normalize=False,
        plus per-axis scales to map back.
        """
        H0, W0 = img_bgr.shape[:2]
        # Resize (no padding)
        img_rs = cv2.resize(img_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.normalize:
            img_rgb /= 255.0
        blob = img_rgb.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

        # per-axis scale factors (model space -> original space uses division by these)
        sx = self.input_w / float(W0)
        sy = self.input_h / float(H0)
        return blob, sx, sy, (W0, H0)

    def predict(self, img_bgr):
        """
        Args:
            img_bgr: HxWx3 uint8 (BGR)
        Returns:
            np.ndarray [M, 6]: [cx, cy, w, h, conf, cls] in ORIGINAL image coordinates (float32)
        """
        blob, sx, sy, (W0, H0) = self._preprocess(img_bgr)

        # ---- inference ----
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        out = np.squeeze(out)  # expect [C+4, N] or [N, C+4]

        # ---- parse output (Ultralytics ONNX variants) ----
        if out.ndim == 2 and out.shape[0] <= out.shape[1]:
            # shape [C+4, N] -> [N, C+4]
            out = out.T

        boxes_cxcywh = out[:, :4]                # (N,4)
        cls_scores = out[:, 4:]                  # (N,num_classes)
        confidences = cls_scores.max(axis=1)     # (N,)
        class_ids = cls_scores.argmax(axis=1)    # (N,)

        # ---- confidence filter ----
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # ---- model-space (480x640) -> original image coords
        boxes_xyxy = self._xywh2xyxy(boxes_cxcywh.copy())
        # de-scale per axis
        boxes_xyxy[:, [0, 2]] /= sx
        boxes_xyxy[:, [1, 3]] /= sy
        # clip to original bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, W0 - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, H0 - 1)

        # ---- size filter in original pixels
        wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        size_mask = (wh[:, 0] >= self.min_w) & (wh[:, 1] >= self.min_h)
        if not np.any(size_mask):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = boxes_xyxy[size_mask]
        confidences = confidences[size_mask]
        class_ids = class_ids[size_mask]

        # ---- NMS (class-agnostic)
        keep = self._nms(boxes_xyxy, confidences, self.nms_threshold)
        if keep.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # ---- xyxy -> cxcywh in original coords
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
        w  = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        h  = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        res = np.stack([cx, cy, w, h, confidences, class_ids.astype(np.float32)], axis=1).astype(np.float32)
        return res


        




if __name__ == "__main__":
    predictor = YoloPredictor(onnx_path="pretrained/yolov8n_visdrone_add_people.onnx",
                              conf_threshold=0.1)
    img = cv2.imread("imgs/frame_yuv/00000169.jpg")
    results = predictor.predict(img)
    img_vis = img.copy()

    CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "people"}
    PALETTE = [
        (37,255,225), (255,191,0), (0,255,0), (0,165,255), (255,0,255),
        (180,105,255), (0,215,255), (255,144,30), (144,238,144), (30,105,210)
    ]
    for cx, cy, w, h, score, cid in results:
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cid = int(cid)
        color = PALETTE[cid % len(PALETTE)]
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cid]}: {score:.2f}"
        cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite("result.jpg", img_vis)


    