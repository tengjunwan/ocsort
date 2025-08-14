import numpy as np


def cxcywh2xyxy(boxes):
    x1y1x2y2 = np.zeros_like(boxes)
    x1y1x2y2[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    x1y1x2y2[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    x1y1x2y2[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    x1y1x2y2[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return x1y1x2y2


def buffer_bbox(bboxes, buffer_ratio):  # (#bboxes, 4), cx, cy, w, h
    buffer_bboxes = np.zeros_like(bboxes)
    buffer_bboxes[:, :2] = bboxes[:, :2]  # cx, cy unchanged
    if isinstance(buffer_ratio, np.ndarray) and len(buffer_bboxes) == len(buffer_ratio):
        buffer_ratio = buffer_ratio.reshape(len(buffer_bboxes), 1)

    buffer_bboxes[:, 2:] = bboxes[:, 2:] * (1 + 2 * buffer_ratio)  # b_w = (1 + 2b) * w, b_h = (1 + 2b) * h 
    return buffer_bboxes


def exp_saturate_by_age(age, vmin=0.2, vmax=1.0, tau=10.0):
    """
    Asymptotically grows from vmin to vmax with e-folding time tau.
    tau=2.5, satuate frame≈8
    tau=5.0, satuate frame≈15
    tau=10.0, satuate frame≈30
    """
    age = np.asarray(age, dtype=np.float32)
    val = vmin + (vmax - vmin) * (1.0 - np.exp(-age / float(tau)))
    return np.clip(val, vmin, vmax)


def iou_batch(bboxesA, bboxesB, buffer_ratioA=0.0, buffer_ratioB=0.0):  # 
    """
    From SORT: Computes IOU between two bboxes in the form [cx,cy,w,h]
        bboxesA: detections, shape = (#dets, 4).
        bboxesB: trackers, shape = (#trks, 4).
    """
    # buffer height and weight
    if isinstance(buffer_ratioA, float) and buffer_ratioA == 0.0:
        bboxesA = bboxesA
    else:
        bboxesA = buffer_bbox(bboxesA, buffer_ratioA)

    if isinstance(buffer_ratioB, float) and buffer_ratioB == 0.0:
        bboxesB = bboxesB
    else:
        bboxesB = buffer_bbox(bboxesB, buffer_ratioB)

    # [cx, cy, w, h] -> [x1, y2, x2, y2]
    bboxesA = cxcywh2xyxy(bboxesA)
    bboxesB = cxcywh2xyxy(bboxesB)

    bboxesB = np.expand_dims(bboxesB, 0)  # (1, #trks, 4)
    bboxesA = np.expand_dims(bboxesA, 1)  # (#dets, 1, 4)
    
    xx1 = np.maximum(bboxesA[..., 0], bboxesB[..., 0])
    yy1 = np.maximum(bboxesA[..., 1], bboxesB[..., 1])
    xx2 = np.minimum(bboxesA[..., 2], bboxesB[..., 2])
    yy2 = np.minimum(bboxesA[..., 3], bboxesB[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h  
    o = wh / ((bboxesA[..., 2] - bboxesA[..., 0]) * (bboxesA[..., 3] - bboxesA[..., 1])                                      
        + (bboxesB[..., 2] - bboxesB[..., 0]) * (bboxesB[..., 3] - bboxesB[..., 1]) - wh)                                              
    return o  # (#dets, #trks)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]  # (#trks, 2, 1)
    CX1 = dets[:,0]  # (#dets)
    CY1 = dets[:,1]  # (#dets)
    CX2 = tracks[:,0]  # (#trks, 1)
    CY2 = tracks[:,1]  # (#trks, 1)
    dx = CX1 - CX2  # (#trks, #dets)
    dy = CY1 - CY2  # (#trks, #dets)
    norm = np.sqrt(dx**2 + dy**2) + 1e-6  # (#trks, #dets)
    dx = dx / norm  # (#trks, #dets)
    dy = dy / norm  # (#trks, #dets)
    return dx, dy # size: num_track x num_det


def appearance_batch(det_feats, trk_feats):
    # normalize feat 
    det_feats_norm = np.linalg.norm(det_feats, axis=1, keepdims=True)  # (#dets, feat_dim)
    det_feats_norm[det_feats_norm == 0] = 1  # avoid division by zero
    det_feats_normalized = det_feats / det_feats_norm  # (#dets, feat_dim)
    trk_feats_norm = np.linalg.norm(trk_feats, axis=1, keepdims=True)  # (#trks, feat_dim)
    trk_feats_norm[trk_feats_norm == 0] = 1  # avoid division by zero
    trk_feats_normalized = trk_feats / trk_feats_norm  # (#trks, feat_dim)

    cos_sim_matrix = det_feats_normalized @ trk_feats_normalized.T  # (#dets, #trks)

    return cos_sim_matrix