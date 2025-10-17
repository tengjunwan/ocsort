import numpy as np


class CoordinateProjector():
    """
    simple coordinate projector to convert between pixel coordinate and world coordinate
    """

    def __init__(self, fx, fy, calibration_w, calibration_h, h=None):
        # fx, fy configurations(given at calibration stage)
        self.fx = fx
        self.fy = fy
        self.calibration_w = calibration_w
        self.calibration_h = calibration_h
        self.h = h         # height above plane ground

        # gimbal status 
        self.gimbal_status_is_initialized = False  # flag
        self.theta = None  # pitch(rad)
        self.phi = None    # yaw(rad)
        self.zoom = None   
        self.fx_for_use = None
        self.fy_for_use = None

        # previous gimbal status
        self.prev_theta = None  # pitch(rad)
        self.prev_phi = None    # yaw(rad)
        self.prev_zoom = None   
        self.prev_fx_for_use = None
        self.prev_fy_for_use = None


        # other status
        self.process_img_shape_is_initialized = False  # flag
        self.process_img_w = None
        self.process_img_h = None
        

    def set_process_img_shape(self, img_w, img_h):
        # set image shape for adjusting fx, fy 
        self.process_img_w = img_w
        self.process_img_h = img_h
        self.process_img_shape_is_initialized = True  # update state

    def init_gimbal_status(self, theta, phi, zoom):
        self.set_gimbal_status(theta, phi, zoom)
        self.set_prev_gimbal_status(theta, phi, zoom)
        self.gimbal_status_is_initialized = True  # update state

    def set_gimbal_status(self, theta, phi, zoom):
        if self.gimbal_status_is_initialized:  # not first time
            self.prev_theta = self.theta
            self.prev_phi = self.phi
            self.prev_zoom = self.zoom
            self.prev_fx_for_use = self.fx_for_use
            self.prev_fy_for_use = self.fy_for_use

        self.theta = theta
        self.phi = phi
        self.zoom = zoom

        # adjusting fx, fy
        if self.process_img_w is None or self.process_img_h is None:
            raise ValueError("need to set resolution of the processing image by calling method 'set_process_img_shape'!")
        self.fx_for_use = self.fx * (self.process_img_w / self.calibration_w) * self.zoom
        self.fy_for_use = self.fy * (self.process_img_h / self.calibration_h) * self.zoom

    def set_prev_gimbal_status(self, theta, phi, zoom):
        self.prev_theta = theta
        self.prev_phi = phi
        self.prev_zoom = zoom

        # adjusting fx, fy
        if self.process_img_w is None or self.process_img_h is None:
            raise ValueError("need to set resolution of the processing image by calling method 'set_process_img_shape'!")
        self.prev_fx_for_use = self.fx * (self.process_img_w / self.calibration_w) * self.prev_zoom
        self.prev_fy_for_use = self.fy * (self.process_img_h / self.calibration_h) * self.prev_zoom


    # def set_zoom(self, zoom):
    #     self.zoom = zoom
    #     # adjusting fx, fy
    #     if self.process_img_w is None or self.process_img_h is None:
    #         raise ValueError("need to set resolution of the processing image by calling method 'set_process_img_shape'!")
    #     self.fx_for_use = self.fx * (self.process_img_w / self.calibration_w) * self.zoom
    #     self.fy_for_use = self.fy * (self.process_img_h / self.calibration_h) * self.zoom

    # def set_pitch_and_yaw(self, theta, phi):
    #     self.theta = theta
    #     self.phi = phi

    def calculate_dphi_and_dtheta(self, dx, dy):
        # dx: pixel translation in x direction(dx = xi+1 - xi) from previous frame to current frame
        # dy: pixel translation in y direction(dy = yi+1 - yi) from previous frame to current frame

        cth = np.cos(self.theta)
        cth = np.where(np.abs(cth) < 1e-6, np.sign(cth) * 1e-6, cth)

        
        d_theta = -(dy / self.fy_for_use)
        d_phi = -(dx / self.fx_for_use) / cth


        return d_phi, d_theta

    def project_from_pixel_to_world(self, pixel_dets):
        """
        Map pixels (absolute, image origin top-left) to world.
        pixel_dets: numpy.ndarray, shape=(#dets, 4 or more), or shape=(4 or more), first 4 columns are xi, yi, wi, hi
        """
        original_shape = pixel_dets.shape
        if len(original_shape) == 1 and original_shape[0] >= 4:   
            pixel_dets = pixel_dets.reshape(1, -1)

        xi = pixel_dets[:, 0]
        yi = pixel_dets[:, 1]
        wi = pixel_dets[:, 2]
        hi = pixel_dets[:, 3]

        # ===part 1: convert cx and cy from pixel to world===
        # shift origin to image center
        xi = xi - 0.5 * self.process_img_w
        yi = yi - 0.5 * self.process_img_h

        # normalized image coords
        u = xi / self.fx_for_use
        v = yi / self.fy_for_use

        cth, sth = np.cos(self.theta), np.sin(self.theta)
        cph, sph = np.cos(self.phi), np.sin(self.phi)

        # plane intersection scale
        denom = cth * v + sth
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        a = self.h / denom

        # initial-world coordinates (axes fixed to first frame)
        xw = a * ( cph * u - sph * sth * v + sph * cth )
        zw = a * (-sph * u - cph * sth * v + cph * cth )

        # ===part 2: convert w and h from pixel to world===
        b = self.h / max(sth, 1e-6)
        ww = wi * b/ self.fx_for_use
        hw = hi * b/ self.fy_for_use


        world_dets = pixel_dets.copy()
        world_dets[:, 0] = xw
        world_dets[:, 1] = zw
        world_dets[:, 2] = ww
        world_dets[:, 3] = hw

        world_dets = world_dets.reshape(original_shape)

        return world_dets
    

    def project_from_world_to_pixel(self, world_dets, use_prev_gimbal=False):
        """
        Map world to pixels (absolute, image origin top-left).
        world_dets: numpy.ndarray, shape=(#dets, 4 or more), or shape=(4 or more), first 4 columns are xw, zw, ww, hw
        """
        original_shape = world_dets.shape
        if len(original_shape) == 1 and original_shape[0] >= 4:   
            world_dets = world_dets.reshape(1, -1)

        xw = world_dets[:, 0]
        zw = world_dets[:, 1]
        ww = world_dets[:, 2]
        hw = world_dets[:, 3]

        # ===prepare===
        if not use_prev_gimbal:  # current gimbal status
            cth, sth = np.cos(self.theta), np.sin(self.theta)
            cph, sph = np.cos(self.phi), np.sin(self.phi)
            fx = self.fx_for_use
            fy = self.fy_for_use
        else:
            cth, sth = np.cos(self.prev_theta), np.sin(self.prev_theta)
            cph, sph = np.cos(self.prev_phi), np.sin(self.prev_phi)
            fx = self.prev_fx_for_use
            fy = self.prev_fy_for_use

        # ===part 1: convert cx and cy from world to pixel===
        yw = self.h
        a = sph * cth * xw + sth * yw + cph * cth * zw
        a = np.where(np.abs(a) < 1e-6, np.sign(a) * 1e-6, a)
        
        xi = (fx / a) * (cph * xw - sph * zw) 
        xi = xi + 0.5 * self.process_img_w  # top left corner as origin
        yi = (fy / a) * (-sph * sth * xw + cth * yw -cph * sth * zw)
        yi = yi + 0.5 * self.process_img_h  # top left corner as origin

        # ===part 2: convert w and h from world to pixel===
        b = sth / max(self.h, 1e-6) 
        wi = b * fx * ww
        hi = b * fy * hw

        pixel_dets = world_dets.copy()
        pixel_dets[:, 0] = xi
        pixel_dets[:, 1] = yi
        pixel_dets[:, 2] = wi
        pixel_dets[:, 3] = hi

        pixel_dets = pixel_dets.reshape(original_shape)

        return pixel_dets
    

    def project_velocity_from_world_to_pixel(self, world_dets, world_velocity):
        """
        Map velocity from world to pixel.
        world_dets: numpy.ndarray, shape=(#dets, 4 or more), or shape=(4 or more), first 4 columns are xw, zw, ww, hw
        world_velocity: numpy.ndarray, shape=(#velocities, 2 or more), or shape=(2 or more), first 2 columns are vxw, vyw
        """
        original_shape = world_dets.shape
        if len(original_shape) == 1 and original_shape[0] >= 4:   
            world_dets = world_dets.reshape(1, -1)

        xw = world_dets[:, 0]
        zw = world_dets[:, 1]

        cth, sth = np.cos(self.theta), np.sin(self.theta)
        cph, sph = np.cos(self.phi), np.sin(self.phi)

        yw = self.h
        a = sph * cth * xw + sth * yw + cph * cth * zw
        a = np.where(np.abs(a) < 1e-6, np.sign(a) * 1e-6, a)
        
        v_original_shape = world_velocity.shape
        if len(v_original_shape) == 1 and v_original_shape[0] >= 2:   
            world_velocity = world_velocity.reshape(1, -1)

        vxw = world_velocity[:, 0]
        vzw = world_velocity[:, 1]

        vxi = (self.fx_for_use / a) * (cph * vxw - sph *vzw)
        vyi = (self.fy_for_use / a) * (-sph * sth * vxw - cph * sth * vzw)

        pixel_velocity = world_velocity.copy()
        pixel_velocity[:, 0] = vxi
        pixel_velocity[:, 1] = vyi

        pixel_velocity = pixel_velocity.reshape(v_original_shape)

        return pixel_velocity


        

        



if __name__ == "__main__":
    import csv
    import shutil
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt

    from kalmanfilterboxnomatrix_new import KalmanFilterBoxTrackerNoMatrix
    from camera_motion_compensate import UnifiedCMC
    from object_detect import YoloPredictor



    # helper function
    def draw_box(frame, cx, cy, w, h, color):
        x1 = int(cx - 0.5 * w)
        y1 = int(cy - 0.5 * h)
        x2 = int(cx + 0.5 * w)
        y2 = int(cy + 0.5 * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


    VIS_DIR = Path("vis_project")
    if VIS_DIR.exists():
        shutil.rmtree(VIS_DIR)
    VIS_DIR.mkdir()

    # load gimbal status
    csv_file = "test_gimbal_data/test_csv/0006_W.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)   # skip header row, if present
        csv_content = list(reader)
    print(f"gimbal status number: {len(csv_content)}")

    # load video paths of frames
    image_dir = Path("test_gimbal_data/test_frames/0006_W")
    img_paths = sorted(list(image_dir.glob("*.jpg")))
    img_num = len(img_paths)

    print(f"Images number: {len(csv_content)}")

    assert img_num == len(csv_content), "something wrong"

    # load target coordinate
    target_coors = np.load("target_coor.npy")

    print(f"target_coors shape: {target_coors.shape}")
    assert len(target_coors) == len(csv_content), "something wrong"

    # load CMC
    cmc = UnifiedCMC(min_features=30, method="optflow", process_img_shape=(240, 320))

    # load detector
    predictor = YoloPredictor(onnx_path="my_script/yolov8n_visdrone_add_people_merged_640_640.onnx",
                              conf_threshold=0.25,
                              nms_threshold=0.7,
                              min_w=10,
                              min_h=10)  # for object detection

    FX = 5773.5
    FY = 5783.2
    CALIBRATION_W = 8000.0
    CALIBRATION_H = 6000.0
    HEIGHT = 165.0

    projector = CoordinateProjector(fx=FX, fy=FY, calibration_w=CALIBRATION_W, calibration_h=CALIBRATION_H, h=HEIGHT)

    all_xw_byMethodA = len(csv_content) * [None]  # len(csv_content)
    all_zw_byMethodA = len(csv_content) * [None]  # len(csv_content)

    all_xw_byMethodB = len(csv_content) * [None]  # len(csv_content)
    all_zw_byMethodB = len(csv_content) * [None]  # len(csv_content)

    all_xw_byMethodC = len(csv_content) * [None]  # len(csv_content)
    all_zw_byMethodC = len(csv_content) * [None]  # len(csv_content)

    kf_box = None  # kalman filter box
    world_coors = np.zeros_like(target_coors)
    predictions = np.zeros((len(csv_content), 4), dtype=np.float32)
    corrections = np.zeros((len(csv_content), 4), dtype=np.float32)
    detections = np.zeros((len(csv_content), 4), dtype=np.float32)

    # method A
    cumu_d_phi_deg = 0.0
    cumu_d_theta_deg = 0.0


    # method C
    cumu_d_phi_deg_corr = 0.0
    cumu_d_theta_deg_corr = 0.0
    phi_deg_error_thresh = 2.0
    theta_deg_error_thresh = 1.0
    for i in range(len(csv_content)):
        if i > 800:
            break
        if i < 15:
            continue  #skip 

        if i == 120:
            print("debug")

        print(f"=========frame-index: {i}=========")

        target_coor = target_coors[i]
        world_coor = world_coors[i]

        # load gimbal status from csv file
        lag_frame = 9
        line = csv_content[i - lag_frame]
        pitch_abs_deg = float(line[1]) * (-1)
        pitch_delta_deg = float(line[2]) * (-1)
        yaw_abs_deg = float(line[3])
        yaw_delta_deg = float(line[4])
        zoom  = float(line[7]) * 3  # need '*3' since logging is somehow not correct
        raser_distance = float(line[8])    

        # load image
        img = cv2.imread(str(img_paths[i]))

        # detect(YOLO)
        det_results = predictor.predict(img)

        # CMC
        curr_affine_matrix = cmc.update(img, det_results)
        dx = curr_affine_matrix[0, 2]
        dy = curr_affine_matrix[1, 2]


        # set resolution
        process_img_h, process_img_w = img.shape[:2]
        projector.set_process_img_shape(process_img_w, process_img_h)

        # set initial gimbal status
        if not projector.gimbal_status_is_initialized:
            projector.init_gimbal_status(theta=np.deg2rad(pitch_abs_deg), phi=np.deg2rad(yaw_abs_deg), zoom=zoom)

        # =========method A: use dφ dθ calculated by Image=========
        d_phi_rad, d_theta_rad = projector.calculate_dphi_and_dtheta(dx, dy)  # radian
        d_phi_deg, d_theta_deg = np.rad2deg(d_phi_rad), np.rad2deg(d_theta_rad)

        cumu_d_phi_deg += d_phi_deg
        cumu_d_theta_deg += d_theta_deg

        # set gimbal status
        projector.set_gimbal_status(theta=np.deg2rad(pitch_abs_deg + cumu_d_theta_deg), phi=np.deg2rad(yaw_abs_deg + cumu_d_phi_deg), zoom=zoom)

        # project
        world_coor = projector.project_from_pixel_to_world(target_coor)
        all_xw_byMethodA[i] = world_coor[0]
        all_zw_byMethodA[i] = world_coor[1]

        print(f"zoom: {zoom:.4f}")
        print(f"dx: {dx:.2f}, dy: {dy:.2f}")
        print(f"calculated cumulative Δφ: {cumu_d_phi_deg:.4f}, cumu calcuated Δθ: {cumu_d_theta_deg:.4f}")
        print(f"loaded cumulative Δφ: {yaw_delta_deg:.4f}, loaded cumulative Δθ: {pitch_delta_deg:.4f}")


        # =========method B: use dφ dθ directly by log=========
        # set gimbal status
        projector.set_gimbal_status(theta=np.deg2rad(pitch_abs_deg + pitch_delta_deg), phi=np.deg2rad(yaw_abs_deg + yaw_delta_deg), zoom=zoom)

        world_coor = projector.project_from_pixel_to_world(target_coor)
        all_xw_byMethodB[i] = world_coor[0]
        all_zw_byMethodB[i] = world_coor[1]


        # =========method C: use dφ dθ calculated by Image, and corrected by Log if accumulated error is big enough=========
        cumu_d_phi_deg_corr += d_phi_deg
        cumu_d_theta_deg_corr += d_theta_deg

        # set gimbal status
        projector.set_gimbal_status(theta=np.deg2rad(pitch_abs_deg + cumu_d_theta_deg_corr), phi=np.deg2rad(yaw_abs_deg + cumu_d_phi_deg_corr), zoom=zoom)

        world_coor = projector.project_from_pixel_to_world(target_coor)
        all_xw_byMethodC[i] = world_coor[0]
        all_zw_byMethodC[i] = world_coor[1]

        # check error is big enough and if big enough, force it to be equal to log value
        d_phi_deg_diff = abs(cumu_d_phi_deg_corr - yaw_delta_deg)  
        d_theta_deg_diff = abs(cumu_d_theta_deg_corr - pitch_delta_deg)
        print(f"cumulative error Δφ: {d_phi_deg_diff:.4f}, cumulative error Δθ: {d_theta_deg_diff:.4f}")

        gimbal_status_is_corrected = False
        if d_phi_deg_diff > phi_deg_error_thresh:
            cumu_d_phi_deg_corr = yaw_delta_deg
            gimbal_status_is_corrected = True
        if d_theta_deg_diff > theta_deg_error_thresh:
            cumu_d_theta_deg_corr = pitch_delta_deg
            gimbal_status_is_corrected = True

        



        # back_pixel_coor = projector.project_from_world_to_pixel(world_coor)

        # print(f"pixel coordinate: (cx: {target_coor[0]:.2f}, cy: {target_coor[1]:.2f}, w: {target_coor[2]:.2f}, h: {target_coor[3]:.2f})")
        # print(f"world coordinate: (cx: {world_coor[0]:.2f}, cy: {world_coor[1]:.2f}, w: {world_coor[2]:.2f}, h: {world_coor[3]:.2f})")
        # # 'back_pixel_coor' should be equal to 'target_coor' 
        # print(f"back pixel coordinate: (cx: {back_pixel_coor[0]:.2f}, cy: {back_pixel_coor[1]:.2f}, w: {back_pixel_coor[2]:.2f}, h: {back_pixel_coor[3]:.2f})")

        # if kf_box is not None:
        #     p_cx, p_cy, p_w, p_h = kf_box.predict()
        #     predictions[i] = np.array([p_cx, p_cy, p_w, p_h])

        # if not np.all(target_coor == 0):  # valid detection(observation)
        #     d_cx, d_cy, d_w, d_h = world_coor
        #     detections[i] = np.array([d_cx, d_cy, d_w, d_h])

        #     if kf_box is None:  # create kalman filter box
        #         kf_box = KalmanFilterBoxTrackerNoMatrix(d_cx, d_cy, d_w, d_h, 5)  # init
                
        #     else:
        #         c_cx, c_cy, c_w, c_h = kf_box.update(np.array([d_cx, d_cy, d_w, d_h], dtype=np.float32))
        #         corrections[i] = np.array([c_cx, c_cy, c_w, c_h])
            
        # else:  # no detection(observation)
        #     c_cx, c_cy, c_w, c_h = kf_box.update(None)
        #     corrections[i] = np.array([c_cx, c_cy, c_w, c_h])


        print("done")

    # end of for loop

    # # Plot world cumulative centers on one image
    # for i in range(800):
    #     # ===method A===
    #     # slice up to i
    #     xws_byMethodA = all_xw_byMethodA[:i+1]
    #     zws_byMethodA = all_zw_byMethodA[:i+1]

    #     # filter out None
    #     xws_valid_byMethodA = [x for x, z in zip(xws_byMethodA, zws_byMethodA) if x is not None and z is not None]
    #     zws_valid_byMethodA = [z for x, z in zip(xws_byMethodA, zws_byMethodA) if x is not None and z is not None]

    #     # ===method B===
    #     # slice up to i
    #     xws_byMethodB = all_xw_byMethodB[:i+1]
    #     zws_byMethodB = all_zw_byMethodB[:i+1]

    #     # filter out None
    #     xws_valid_byMethodB = [x for x, z in zip(xws_byMethodB, zws_byMethodB) if x is not None and z is not None]
    #     zws_valid_byMethodB = [z for x, z in zip(xws_byMethodB, zws_byMethodB) if x is not None and z is not None]

    #     # ===method C===
    #     # slice up to i
    #     xws_byMethodC = all_xw_byMethodC[:i+1]
    #     zws_byMethodC = all_zw_byMethodC[:i+1]

    #     # filter out None
    #     xws_valid_byMethodC = [x for x, z in zip(xws_byMethodC, zws_byMethodC) if x is not None and z is not None]
    #     zws_valid_byMethodC = [z for x, z in zip(xws_byMethodC, zws_byMethodC) if x is not None and z is not None]


    #     plt.figure()
    #     plt.plot(xws_byMethodA, zws_byMethodA, 'o-', color='blue', markersize=3)
    #     plt.plot(xws_byMethodB, zws_byMethodB, 'o-', color='red', markersize=3)
    #     plt.plot(xws_byMethodC, zws_byMethodC, 'o-', color='green', markersize=3)
    #     plt.xlabel("Xw (m)")
    #     plt.ylabel("Zw (m)")
    #     plt.title("Projected world trajectory of target car")
    #     plt.axis("equal")
    #     plt.grid(True)
    #     plt.savefig(str(VIS_DIR / f"world_trajectory_{i}.png"), dpi=200)
    #     plt.close()

    
    # ===method A===
    # slice up to i
    xws_byMethodA = all_xw_byMethodA[100:110]
    zws_byMethodA = all_zw_byMethodA[100:110]

    # filter out None
    xws_valid_byMethodA = [x for x, z in zip(xws_byMethodA, zws_byMethodA) if x is not None and z is not None]
    zws_valid_byMethodA = [z for x, z in zip(xws_byMethodA, zws_byMethodA) if x is not None and z is not None]

    plt.figure()
    plt.plot(xws_byMethodA, zws_byMethodA, 'o-', color='blue', markersize=3)
    plt.xlabel("Xw (m)")
    plt.ylabel("Zw (m)")
    plt.title("Projected world trajectory of target car")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(str(VIS_DIR / f"world_trajectory_700to800_cal.png"), dpi=200)
    plt.close()

    # ===method B===
    # slice up to i
    xws_byMethodB = all_xw_byMethodB[100:110]
    zws_byMethodB = all_zw_byMethodB[100:110]

    # filter out None
    xws_valid_byMethodB = [x for x, z in zip(xws_byMethodB, zws_byMethodB) if x is not None and z is not None]
    zws_valid_byMethodB = [z for x, z in zip(xws_byMethodB, zws_byMethodB) if x is not None and z is not None]

    plt.figure()
    plt.plot(xws_byMethodB, zws_byMethodB, 'o-', color='red', markersize=3)
    plt.xlabel("Xw (m)")
    plt.ylabel("Zw (m)")
    plt.title("Projected world trajectory of target car")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(str(VIS_DIR / f"world_trajectory_700to800_log.png"), dpi=200)
    plt.close()

    # ===method C===
    # slice up to i
    xws_byMethodC = all_xw_byMethodC[100:110]
    zws_byMethodC = all_zw_byMethodC[100:110]

    # filter out None
    xws_valid_byMethodC = [x for x, z in zip(xws_byMethodC, zws_byMethodC) if x is not None and z is not None]
    zws_valid_byMethodC = [z for x, z in zip(xws_byMethodC, zws_byMethodC) if x is not None and z is not None]

    plt.figure()
    plt.plot(xws_byMethodC, zws_byMethodC, 'o-', color='green', markersize=3)
    plt.xlabel("Xw (m)")
    plt.ylabel("Zw (m)")
    plt.title("Projected world trajectory of target car")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(str(VIS_DIR / f"world_trajectory_700to800_corr.png"), dpi=200)
    plt.close()

  

    # # visualize kalman filter 
    # xws_valid = [x for x, z in zip(all_xw, all_zw) if x is not None and z is not None]
    # zws_valid = [z for x, z in zip(all_xw, all_zw) if x is not None and z is not None]


    # # xws_valid = xws_valid[:20]
    # # zws_valid = zws_valid[:20]

    # xws_max = max(xws_valid)
    # xws_min = min(xws_valid)
    # zws_max = max(zws_valid)
    # zws_min = min(zws_valid)
    
    # pad = 10

    # canvas_w = int(xws_max - xws_min + 2 * pad)
    # canvas_h = int(zws_max - zws_min + 2 * pad)

    
    # size_scale = 30
    # for i in range(len(csv_content)):
    #     if i < 15:
    #         continue
    #     canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    #     # draw prediction
    #     pred = predictions[i]
    #     if not np.all(pred == 0):  # valid pred
    #         draw_box(canvas, pred[0] - xws_min + pad, pred[1] - zws_min + pad, pred[2] * size_scale, pred[3] * size_scale, (255,0,0))  # blue

    #     # draw corrrection
    #     corr = corrections[i]
    #     if not np.all(corr == 0):  # valid corr
    #         draw_box(canvas, corr[0] - xws_min + pad, corr[1] - zws_min + pad, corr[2] * size_scale, corr[3] * size_scale, (0,0,255))  # red

    #     # draw detection
    #     det = detections[i]
    #     if not np.all(det == 0):  # valid det
    #         draw_box(canvas, det[0] - xws_min + pad, det[1] - zws_min + pad, det[2] * size_scale, det[3] * size_scale, (0,0,255))  # red

    #     # Process the frame
    #     cv2.imwrite(VIS_DIR / f"{i:07d}.jpg", canvas)


        


        
         


        

    
