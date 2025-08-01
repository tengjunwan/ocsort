import onnxruntime
import numpy as np
import cv2


from oc_sort import iou_batch, linear_assignment


class ReID():
    """Basic REID model for calculating cosine similarity"""

    def __init__(self, onnx_path="my_script/fastreid_model_b1_s128.onnx", feat_dim=128, input_size=128):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean = np.array([123.6750, 116.2800, 103.5300], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([58.3950, 57.1200, 57.3750], dtype=np.float32).reshape(1, 3, 1, 1)

    def embed(self, img, need_norm=True):
        # preprocess
        resize_img = cv2.resize(img, (self.input_size, self.input_size))
        input_img = resize_img[:, :, ::-1]  # BGR to RGB
        input_img = input_img.astype(np.float32) 
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0)  # add batch dimension (1,3,H,W)

        # normalize
        if need_norm:
            input_img = (input_img - self.mean) / self.std

        # inference
        output = self.session.run([self.output_name], {self.input_name: input_img})[0]  # (1, feat_num)
        output = np.squeeze(output)
        
        return output
    


class EfficientReIDStrategy():
    """An efficient strategy to implement ReID: selectively embed detected areas(N dets -> M embeddings, M < N)"""

    def __init__(self, maximum_num_per_frame=4, iou_threshold=0.3, det_thresh=0.7, **kwargs):
        self.M = maximum_num_per_frame
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh

        self.embedded_dets = np.empty((0, 4)) # to keep track of embedded dets in "_embed_strategy_tracked_mode"
        

    def select(self, dets, target_position, is_target_tracked=True):
        """
        Args:
            dets(numpy.ndarray): detections given by YOLO. 
            target_position(numpy.ndarray): format [cx, cy], the position of the target.
            is_target_tracked(bool): if the target is tracked or not.
        Return:
            det_feats(numpy.ndarray): shape (#dets, self.reid_model.feat_dim), the embeddings for the dets.
        """

        if is_target_tracked:
            selective_det_idx = self._embed_strategy_tracked_mode(dets, target_position)  # selective index of dets to do embedding
        else:
            selective_det_idx = self._embed_strategy_untracked_mode(dets, target_position)

        return selective_det_idx

    def _embed_strategy_tracked_mode(self, dets, target_position):
        """we gradually embed M dets from near to far each time"""
        # if number of  dets is small already
        if len(dets) <= self.M:
            self.embedded_dets = np.empty((0, 4))
            return np.arange(len(dets))
        
        # calculate distances 
        distance = batch_distance(target_position.reshape(1, -1), dets[:, :2])[0]  # (#dets, )
        low_score_mask = dets[:, 4] < self.det_thresh
        distance[low_score_mask] = distance[low_score_mask] + 1000  # add a large basic distance to low score dets

        # if no embedded_dets, choose the top M nearest dets
        if len(self.embedded_dets) == 0:
            top_M_idx = np.argsort(distance)[:self.M]
            self.embedded_dets = dets[top_M_idx]  # update the embedded dets
            return top_M_idx
            
        # len(dets) > M and len(self.embedded_dets) > 0
        # match current dets with embedded dets in history
        iou = iou_batch(dets[:, :4], self.embedded_dets[:, :4])
        matched_indices = linear_assignment(-iou)
        unmatched_dets = []
        for d, det in enumerate(dets):
            if(d not in matched_indices[:,0]):
                unmatched_dets.append(d)
        matches = []
        for m in matched_indices:
            if iou[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        unmatched_dets = np.array(unmatched_dets, dtype=np.int64)


        if len(unmatched_dets) == self.M:  # case 1: explore from near to distant and just startover
            self.embedded_dets = np.empty((0, 4))
            top_M_idx = unmatched_dets
            return top_M_idx
        
        if len(unmatched_dets) > self.M:  # case 2: still explore from near to distant
            # find top M nearest among unmatched dets
            distance_unmatched = distance[unmatched_dets]
            top_M_idx_unmatched = np.argsort(distance_unmatched)[:self.M]
            top_M_idx = unmatched_dets[top_M_idx_unmatched]
            embedded_det_idx = np.concatenate([matches[:, 0], top_M_idx], dtype=np.int64)
            self.embedded_dets = dets[embedded_det_idx]
            return top_M_idx
            
        if len(unmatched_dets) == 0:  # case 4: rare case, however it could happen if detection is not stable
            top_M_idx = np.argsort(distance)[:self.M]
            self.embedded_dets = dets[top_M_idx]
            return top_M_idx
        
        if len(unmatched_dets) < self.M:  # case 3: explore from near to distant and startover and more
            top_idx = np.argsort(distance)[: self.M - len(unmatched_dets)]
            top_M_idx = np.concatenate([unmatched_dets, top_idx])
            self.embedded_dets = dets[top_idx]
            return top_M_idx


    def _embed_strategy_untracked_mode(self, dets, target_position):
        """we simply only want to know the top M nearest dets close to target"""
        if len(dets) <= self.M:
            return np.arange(len(dets))

        distance = batch_distance(target_position.reshape(1, -1), dets[:, :2])[0]  # (#dets, )
        top_M_idx = np.argsort(distance)[:self.M]
        return top_M_idx
        


def batch_distance(pts1, pts2):
    """
    Compute pairwise Euclidean distances between two sets of 2D points.
    
    Args:
        pts1: ndarray of shape (N, 2)
        pts2: ndarray of shape (M, 2)
    
    Returns:
        distances: ndarray of shape (N, M), where distances[i, j] is the distance
                   between pts1[i] and pts2[j]
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]  # shape (N, M, 2)
    dists = np.linalg.norm(diff, axis=2)  # shape (N, M)
    return dists
    


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3, linewidth=150)
    embedder = ReID()
    need_norm = True
    img1 = cv2.imread("test_imgs/reid/cropResized_1_0_128_128.jpg")  # red bus
    feat1 = embedder.embed(img1, need_norm)
    print(f"feat1:\n {feat1}")
    img2 = cv2.imread("test_imgs/reid/cropResized_1_1_128_128.jpg")  # white car
    feat2 = embedder.embed(img2, need_norm)
    print(f"feat2:\n {feat2}")
    img3 = cv2.imread("test_imgs/reid/cropResized_1_3_128_128.jpg")  # white car
    feat3 = embedder.embed(img3, need_norm)
    print(f"feat3:\n {feat3}")

    cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    print(f"similarity score(red bus vs white car): {cos_sim:.2f}")

    cos_sim = np.dot(feat1, feat3) / (np.linalg.norm(feat1) * np.linalg.norm(feat3))
    print(f"similarity score(red bus vs white car): {cos_sim:.2f}")
    # print("done")

    cos_sim = np.dot(feat2, feat3) / (np.linalg.norm(feat2) * np.linalg.norm(feat3))
    print(f"similarity score(white car vs white car): {cos_sim:.2f}")

    feat_3_model = np.array([0.006233, 0.001498, -0.007702, 0.001781, -0.005318, -0.007233, -0.001363, -0.006031, -0.000627, 
                             -0.006001, -0.001946, 0.013199, 0.000000, -0.002760, -0.013199, 0.001823, -0.007496, 0.000685, 
                             -0.000000, -0.005573, 0.002726, -0.018677, 0.008186, 0.007771, -0.003275, 0.001083, 0.002481, 
                             -0.011711, -0.007519, -0.007160, -0.000000, -0.006275, 0.002222, 0.003693, -0.000698, -0.003300, 
                             0.021896, -0.000892, 0.000507, 0.002674, 0.000319, -0.006203, -0.000887, -0.000000, -0.000000, 
                             0.001651, -0.005775, 0.013885, 0.001180, 0.002195, -0.000806, 0.001937, -0.001497, -0.003525, 
                             0.001914, -0.000000, 0.005772, 0.000000, -0.002357, 0.001435, -0.000766, -0.004173, 0.001498, 
                             -0.008209, -0.012505, 0.001299, 0.007267, 0.003052, -0.015205, 0.003830, 0.001449, -0.006107, 
                             0.001391, 0.000085, 0.003788, 0.010361, 0.000856, -0.000000, -0.000789, 0.006542, -0.000164, 
                             -0.001772, 0.001818, 0.002146, -0.000536, 0.018509, -0.005424, 0.000000, -0.000000, 0.001241, 
                             0.000000, 0.003456, -0.001953, 0.014175, -0.006924, 0.004620, 0.000000, -0.000954, -0.010498, 
                             0.000000, -0.000579, 0.006973, 0.008667, 0.000176, 0.004066, 0.000000, -0.000187, 0.002535, 
                             -0.003687, 0.001693, -0.006462, -0.001136, -0.002676, 0.003746, -0.001050, 0.006599, 0.000000, 
                             -0.000000, 0.001648, -0.005386, 0.003616, 0.005436, 0.009895, 0.010178, -0.004272, -0.011559, 
                             0.001307, 0.002703], dtype=np.float32)
    

    feat_2_model = np.array([0.001181, 0.001252, -0.005772, 0.000000, -0.014091, -0.005295, 0.000615, 0.002264, -0.000180, 
                             -0.009995, -0.000576, 0.009415, 0.000000, 0.001781, -0.006260, 0.004917, -0.003508, 0.000630, 
                             -0.000000, -0.004364, 0.005188, -0.014503, 0.002769, 0.003700, -0.000615, -0.005852, 0.002485, 
                             -0.009872, 0.003832, -0.014259, -0.000000, -0.000660, 0.007095, -0.004650, 0.004303, -0.000423, 
                             0.027069, -0.000522, -0.006268, 0.000945, 0.000162, -0.009247, 0.006763, -0.000000, -0.000000, 
                             0.000467, -0.002718, 0.009163, 0.001509, 0.003830, 0.000716, 0.001511, -0.001209, -0.001961, 
                             0.000468, -0.000000, 0.005005, -0.000000, -0.009399, 0.005745, 0.000380, -0.009781, 0.005997, 
                             -0.007637, -0.008102, 0.004025, 0.005062, 0.000551, -0.019608, -0.004513, 0.023590, 0.003042, 
                             -0.000801, 0.000000, 0.004921, 0.009277, 0.000425, 0.000000, 0.005096, 0.002722, -0.003452, 
                             -0.001108, 0.008148, -0.002598, -0.000536, 0.027527, -0.003420, -0.000000, -0.000000, -0.001721, 
                             -0.000000, 0.000193, -0.000695, 0.016754, -0.004738, -0.001837, 0.000000, 0.007099, 0.001450, 
                             -0.000000, -0.005051, 0.003080, 0.017334, -0.001002, 0.015869, 0.000000, -0.000113, -0.001324, 
                             -0.017776, -0.004372, -0.006134, 0.000096, -0.005028, 0.019012, -0.006908, -0.000078, 0.000000, 
                             -0.000000, -0.000301, -0.003096, 0.001432, 0.006546, 0.004436, -0.002193, -0.001590, -0.006195, 
                             -0.002438, -0.001949], dtype=np.float32)
    
    feat_1_model = np.array([-0.010948, 0.000543, -0.004204, 0.004471, 0.073853, 0.012100, -0.015152, 0.052551, 0.015556, 
                             0.006317, 0.001950, -0.049957, 0.000187, -0.001044, 0.023331, -0.004894, -0.034302, 0.001067, 
                             0.000000, -0.020554, 0.036194, -0.032349, 0.033112, 0.077209, -0.043884, 0.043304, -0.038422, 
                             0.097839, 0.048279, -0.074219, -0.000000, -0.088989, -0.059479, -0.042572, 0.038971, -0.008675, 
                             0.055542, 0.001356, 0.010353, -0.014702, 0.002142, -0.011955, 0.031738, -0.000000, -0.000000, 
                             -0.000637, 0.015717, -0.024872, -0.027618, -0.025436, 0.005241, 0.001485, 0.019821, 0.020554, 
                             -0.004097, -0.000000, -0.031860, 0.000375, -0.044556, -0.001579, -0.046844, 0.060211, 0.028381, 
                             -0.017029, 0.009743, -0.021500, -0.041534, -0.019073, -0.080444, 0.005100, -0.031281, 0.066467, 
                             0.004974, -0.000216, -0.003342, 0.027298, 0.002485, 0.000000, 0.082214, -0.013229, -0.020432, 
                             -0.002672, 0.024948, -0.004604, 0.003363, 0.003212, 0.018036, 0.000803, -0.000000, 0.037445, 
                             -0.000256, -0.017441, 0.023056, -0.053711, 0.016983, 0.003084, 0.000000, -0.081482, 0.037659, 
                             -0.000000, -0.031464, 0.004120, 0.004593, 0.029755, 0.046021, 0.000000, -0.001220, -0.030777, 
                             -0.033630, 0.031082, 0.076477, -0.002329, 0.032288, -0.010979, -0.072388, -0.037506, -0.000180, 
                             -0.000000, 0.046082, -0.005775, -0.025467, 0.008667, 0.044830, 0.055756, 0.017365, -0.031189, 
                             0.009102, -0.004967], dtype=np.float32)

    cos_sim = np.dot(feat_1_model, feat_2_model) / (np.linalg.norm(feat_1_model) * np.linalg.norm(feat_2_model))
    print(f"similarity score(red bus vs white car): {cos_sim:.2f}")

    cos_sim = np.dot(feat_1_model, feat_3_model) / (np.linalg.norm(feat_1_model) * np.linalg.norm(feat_3_model))
    print(f"similarity score(red bus vs white car): {cos_sim:.2f}")

    cos_sim = np.dot(feat_2_model, feat_3_model) / (np.linalg.norm(feat_2_model) * np.linalg.norm(feat_3_model))
    print(f"similarity score(white car vs white car): {cos_sim:.2f}")

    # test ReID strategy
    # from pathlib import Path
    # import shutil

    # from tqdm import tqdm
    # import cv2
    
    # from object_detect import YoloPredictor
    
    # img_dir = Path("test_imgs/DJI_20250606155437_0007_V/DJI_20250606155437_0007_V_white_van")
    # img_paths = sorted(img_dir.glob("*.jpg"))
    # img_num = len(img_paths)

    # vis_dir = Path("vis_reid_strategy")
    # if vis_dir.exists():
    #     shutil.rmtree(vis_dir)
    # vis_dir.mkdir(exist_ok=True, parents=True)


    # predictor = YoloPredictor()
    # select_num = 4
    # reid_strategy = EfficientReIDStrategy(select_num)


    # for img_idx, img_path in tqdm(enumerate(img_paths), total=img_num):

    #     img = cv2.imread(img_path)

    #     # YOLO detection
    #     dets = predictor.predict(img)

    #     h, w = img.shape[:2]
    #     target_position = np.array([w / 2.0, h / 2.0])
    #     dets_idx = reid_strategy.select(dets, target_position, True)

    #     assert len(dets_idx) == min(select_num, len(dets)), "something wrong"

    #     # ==================draw==================
    #     img_vis_strategy = img.copy()
    #     for i, (cx, cy, w, h, score) in enumerate(dets):
    #         if i in dets_idx:
    #             color = (0, 255, 0)  # green: currrent pick for ReID
    #         else:
    #             color = (0, 0, 255)  # red

    #         x1 = int(cx - 0.5 * w)
    #         y1 = int(cy - 0.5 * h)
    #         x2 = int(cx + 0.5 * w)
    #         y2 = int(cy + 0.5 * h)
    #         cv2.rectangle(img_vis_strategy, (x1, y1), (x2, y2), 
    #                     color, 1)
    #         label = f" {score:.2f}"
    #         cv2.putText(img_vis_strategy, label, (x2 - 40, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
    #                     0.5, color, 2)
            
    #     cv2.imwrite(vis_dir / img_path.name, img_vis_strategy)





