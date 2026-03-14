import os
import sys
import cv2
import time
import math
import json
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, QFormLayout, 
                             QGroupBox, QSlider, QTextEdit, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# ==========================================
# [融合] 双目相机标定类 (来自 OpenCV光流法 + 双目立体视觉.py)
# ==========================================
class StereoCalibrator:
    def __init__(self):
        self.calibrated = False
        self.K1 = self.K2 = self.D1 = self.D2 = None
        self.R = self.T = self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None
        
    def load_calibration(self, filename):
        try:
            data = np.load(filename)
            self.Q = data['Q']  # 视差转深度矩阵
            self.calibrated = True
            return True
        except:
            return False

# ==========================================
# [融合] 3D 状态平滑与预测 (来自 基于单目相机视觉.py / app.py)
# ==========================================
class KalmanTracker3D:
    def __init__(self, dt=1.0/30.0):
        self.kf = cv2.KalmanFilter(6, 3)
        self.is_initialized = False
        self.missed_frames = 0
        self.max_missed_frames = 15
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0], [0, 1, 0, 0,  dt, 0], [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0], [0, 0, 0, 0,  1,  0], [0, 0, 0, 0,  0,  1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.array([[1e-1, 0, 0], [0, 1e-1, 0], [0, 0, 2e-1]], np.float32)

    def update(self, x, y, z):
        meas = np.array([[x], [y], [z]], np.float32)
        if not self.is_initialized:
            self.kf.statePre = np.array([[x], [y], [z], [0], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [z], [0], [0], [0]], np.float32)
            self.is_initialized = True
            self.missed_frames = 0
            return x, y, z
        self.kf.predict()
        est = self.kf.correct(meas)
        self.missed_frames = 0
        return float(est[0,0]), float(est[1,0]), float(est[2,0])

    def predict_only(self):
        if not self.is_initialized: return None
        self.missed_frames += 1
        if self.missed_frames > self.max_missed_frames:
            self.is_initialized = False 
            return None
        pred = self.kf.predict()
        return float(pred[0,0]), float(pred[1,0]), float(pred[2,0])

# ==========================================
# [融合] LK光流测速核心类 (来自 测无人机速度-LK光流算法.py)
# ==========================================
class LKSpeedEstimator:
    def __init__(self, window_size=8):
        self.tracks = {}  # 存储 track_id -> {'pts': points, 'vel_buf': deque}
        self.window_size = window_size

    def update_speed(self, tid, frame_gray, bbox, distance_meters, fps, fx):
        if tid not in self.tracks:
            self.tracks[tid] = {'pts': None, 'vel_buf': deque(maxlen=self.window_size), 'prev_gray': None}
        
        tracker = self.tracks[tid]
        x, y, w, h = [int(v) for v in bbox]

        # 初始化或重置特征点
        if tracker['pts'] is None or tracker['prev_gray'] is None:
            self._detect_features(tracker, frame_gray, x, y, w, h)
            tracker['prev_gray'] = frame_gray.copy()
            return sum(tracker['vel_buf'])/len(tracker['vel_buf']) if tracker['vel_buf'] else 0.0, None

        # LK 光流计算
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            tracker['prev_gray'], frame_gray, tracker['pts'], None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_new = next_pts[status == 1] if next_pts is not None else []
        good_old = tracker['pts'][status == 1] if next_pts is not None else []

        real_velocity = 0.0
        if len(good_new) < 5:
            # 特征点丢失，重新检测
            self._detect_features(tracker, frame_gray, x, y, w, h)
            real_velocity = sum(tracker['vel_buf']) / len(tracker['vel_buf']) if tracker['vel_buf'] else 0.0
        else:
            # 物理速度换算: V = (像素位移 * FPS * 真实距离) / 焦距
            displacements = np.linalg.norm(good_new - good_old, axis=1)
            avg_pixel_dist = np.mean(displacements)
            instant_velocity = (avg_pixel_dist * fps * distance_meters) / fx
            
            tracker['vel_buf'].append(instant_velocity)
            real_velocity = sum(tracker['vel_buf']) / len(tracker['vel_buf'])
            tracker['pts'] = good_new.reshape(-1, 1, 2)

        tracker['prev_gray'] = frame_gray.copy()
        return real_velocity, good_new

    def _detect_features(self, tracker, gray, x, y, w, h):
        img_h, img_w = gray.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0: return
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.1, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
            tracker['pts'] = pts

    def remove_track(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

# ==========================================
# 核心视频处理与综合分析线程
# ==========================================
class VideoProcessorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()
    init_video_signal = pyqtSignal(int, int)
    update_progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str, str)
    coord_signal = pyqtSignal(dict)  # 传递目标综合信息

    def __init__(self):
        super().__init__()
        self.running, self.is_paused = False, False
        self.video_path, self.req_seek_frame = "", -1
        
        # 统一全系统参数
        self.params = {
            'FX': 1200.0, 'FY': 1200.0, 'CX': 640.0, 'CY': 360.0, 
            'DRONE_W': 0.4, 'DRONE_H': 0.15, 'CAMERA_H': 2.0,
            'X_MIN': -20.0, 'X_MAX': 20.0, 'Z_MIN': 5.0, 'Z_MAX': 100.0,
            'CAM_LAT': 39.904200, 'CAM_LON': 116.407400, 'CAM_ALT': 50.0, 
            'CAM_YAW': 0.0, 'CAM_PITCH': -15.0
        }
        
        self.gov_nfz_polygon_gps = []
        self.enable_local_zone = True 

        # YOLOv11 (或 v8)
        self.model = YOLO("yolov8n.pt") # 使用基础模型，如有自训练模型可替换
        self.trackers = {}
        self.speed_estimator = LKSpeedEstimator()
        
        self.stereo = StereoCalibrator() # 双目预留接口
        self.vision_mode = "Mono" # Mono 或 Stereo

    # [融合] auodrone.py 坐标转换系统 - 基于真实俯仰偏航的 3D 转换
    def _get_3d_world_coords(self, u, v, Z_depth):
        p = self.params
        x_n = (u - p['CX']) / p['FX']
        y_n = (v - p['CY']) / p['FY']
        
        Xc, Yc, Zc = x_n * Z_depth, y_n * Z_depth, Z_depth

        alpha = np.radians(90.0 - p['CAM_PITCH'])
        psi_rad = np.radians(p['CAM_YAW'])

        R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        Rz = np.array([[np.cos(psi_rad), np.sin(psi_rad), 0], [-np.sin(psi_rad), np.cos(psi_rad), 0], [0, 0, 1]])
        
        R = Rz @ Rx @ R0
        delta = R @ np.array([Xc, Yc, Zc])
        
        X_w = delta[0]
        Y_w = delta[1]
        Z_w = p['CAMERA_H'] + delta[2] # 真实相对高度
        
        return X_w, Y_w, Z_w

    def _get_absolute_gps(self, X_w, Y_w):
        # 基于 ENU 坐标系的 GPS 偏移换算
        lat_offset = Y_w / 111320.0
        lon_offset = X_w / (111320.0 * math.cos(math.radians(self.params['CAM_LAT'])))
        return self.params['CAM_LAT'] + lat_offset, self.params['CAM_LON'] + lon_offset

    def _gps_to_relative_2d(self, target_lat, target_lon):
        cam_lat, cam_lon, yaw = self.params['CAM_LAT'], self.params['CAM_LON'], math.radians(self.params['CAM_YAW'])
        north_offset = (target_lat - cam_lat) * 111320.0
        east_offset = (target_lon - cam_lon) * (111320.0 * math.cos(math.radians(cam_lat)))
        return (east_offset * math.cos(yaw) - north_offset * math.sin(yaw),
                east_offset * math.sin(yaw) + north_offset * math.cos(yaw))

    def run(self):
        self.running, self.is_paused, self.req_seek_frame = True, False, -1
        self.trackers.clear()
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit()
            return
            
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.init_video_signal.emit(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps)
        
        while self.running and cap.isOpened():
            if self.req_seek_frame != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.req_seek_frame)
                self.req_seek_frame = -1
                self.trackers.clear()
                
            if self.is_paused:
                time.sleep(0.05); continue

            success, frame = cap.read()
            if not success: break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_sec = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) // fps
            time_str = f"{current_sec // 60:02d}:{current_sec % 60:02d}"
            
            # YOLOv11 Tracking
            results = self.model.track(frame, persist=True, conf=0.25, verbose=False)
            
            active_ids, drone_list = [], [] 
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box, tid in zip(results[0].boxes.xywh.cpu().numpy(), results[0].boxes.id.int().cpu().numpy()):
                    tid = int(tid)
                    active_ids.append(tid)
                    cx_p, cy_p, w_p, h_p = box
                    
                    # 1. 深度估计 (融合 单目先验 vs 双目视差)
                    if self.vision_mode == "Stereo" and self.stereo.calibrated:
                        # 此处为双目伪代码：如果有双目输入，通过 disparity 计算深度
                        Z_depth = 10.0 # 模拟双目解算结果
                    else:
                        # 单目相似三角形法
                        Z_depth = (self.params['DRONE_W'] * self.params['FX']) / max(w_p, 1.0)

                    # 2. 3D 坐标系转换 (融合 auodrone.py)
                    X_w, Y_w, Z_w = self._get_3d_world_coords(cx_p, cy_p, Z_depth)

                    # 3. 卡尔曼平滑 (融合 EKF)
                    if tid not in self.trackers: self.trackers[tid] = KalmanTracker3D()
                    smooth_X, smooth_Y, smooth_Z = self.trackers[tid].update(X_w, Y_w, Z_w)

                    # 4. LK光流测速 (融合 测无人机速度.py)
                    speed, lk_pts = self.speed_estimator.update_speed(
                        tid, frame_gray, (cx_p - w_p/2, cy_p - h_p/2, w_p, h_p), smooth_Z, fps, self.params['FX']
                    )

                    # 5. 防区与地理围栏判断
                    is_local = self.params['X_MIN'] <= smooth_X <= self.params['X_MAX'] and self.params['Z_MIN'] <= smooth_Z <= self.params['Z_MAX']
                    is_gov = False
                    if len(self.gov_nfz_polygon_gps) >= 3:
                        local_pts = [self._gps_to_relative_2d(lat, lon) for lat, lon in self.gov_nfz_polygon_gps]
                        is_gov = cv2.pointPolygonTest(np.array(local_pts, dtype=np.float32), (float(smooth_X), float(smooth_Z)), False) >= 0

                    drone_list.append({
                        'id': tid, 'x': smooth_X, 'y': smooth_Y, 'z': smooth_Z, 
                        'cx': cx_p, 'cy': cy_p, 'w': w_p, 'h': h_p, 'speed': speed,
                        'local': is_local, 'gov': is_gov, 'pts': lk_pts
                    })

            # 清理丢失的目标
            for tid in list(self.trackers.keys()):
                if tid not in active_ids:
                    pred = self.trackers[tid].predict_only()
                    if pred is None:
                        del self.trackers[tid]
                        self.speed_estimator.remove_track(tid)

            # 画面绘制与数据发送
            closest_drone, min_z = None, 9999.0
            for d in drone_list:
                d_color = (0, 140, 255) if d['gov'] else ((0, 0, 255) if d['local'] else (0, 255, 0))
                
                # 画框
                x1, y1 = int(d['cx'] - d['w']/2), int(d['cy'] - d['h']/2)
                x2, y2 = int(d['cx'] + d['w']/2), int(d['cy'] + d['h']/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), d_color, 2)
                
                # 画信息标签 (包含距离、高度、速度)
                label = f"ID:{d['id']} D:{d['z']:.1f}m H:{d['y']:.1f}m V:{d['speed']:.1f}m/s"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, d_color, 2)
                
                # 画光流特征点
                if d['pts'] is not None:
                    for p in d['pts']:
                        cv2.circle(frame, (int(p[0][0]), int(p[0][1])), 3, (255, 0, 255), -1)

                if d['z'] < min_z: min_z, closest_drone = d['z'], d

            if closest_drone:
                abs_lat, abs_lon = self._get_absolute_gps(closest_drone['x'], closest_drone['z'])
                info = {
                    'has_drone': True, 'lat': abs_lat, 'lon': abs_lon, 'alt': self.params['CAM_ALT'] + closest_drone['y'],
                    'x': closest_drone['x'], 'y': closest_drone['y'], 'z': closest_drone['z'], 'speed': closest_drone['speed']
                }
                self.coord_signal.emit(info)
            else:
                self.coord_signal.emit({'has_drone': False})

            # 更新雷达与主画面
            radar_img = self.create_radar_map(drone_list)
            self.change_pixmap_signal.emit(frame, radar_img)
            self.update_progress_signal.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), time_str)

        cap.release()
        self.finished_signal.emit()

    def create_radar_map(self, drone_list):
        radar_size = 400 
        radar_img = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        p = self.params
        
        dynamic_max_z = max(p['Z_MAX'] * 1.2, 50.0) 
        dynamic_max_x = max(max(abs(p['X_MAX']), abs(p['X_MIN'])) * 1.5, 20.0)
        center_u, bottom_v = int(radar_size / 2), radar_size
        
        cv2.line(radar_img, (center_u, 0), (center_u, bottom_v), (50, 50, 50), 1)
        for r in range(1, 6):
            radius = int(r * (radar_size / 5))
            cv2.circle(radar_img, (center_u, bottom_v), radius, (50, 50, 50), 1)
            
        def map_to_radar(x, z):
            return (int(center_u + (x / dynamic_max_x) * (radar_size / 2)),
                    int(bottom_v - (z / dynamic_max_z) * radar_size))
            
        if self.enable_local_zone:
            cv2.rectangle(radar_img, map_to_radar(p['X_MIN'], p['Z_MAX']), map_to_radar(p['X_MAX'], p['Z_MIN']), (0, 150, 0), 2)
        
        nfz_pts_2d = [map_to_radar(*self._gps_to_relative_2d(lat, lon)) for lat, lon in self.gov_nfz_polygon_gps]
        if len(nfz_pts_2d) >= 3:
            pts = np.array([nfz_pts_2d], np.int32)
            overlay = radar_img.copy()
            cv2.fillPoly(overlay, pts, (0, 100, 255)) 
            cv2.addWeighted(overlay, 0.4, radar_img, 0.6, 0, radar_img)
            cv2.polylines(radar_img, pts, True, (0, 140, 255), 2)
            
        for d in drone_list:
            drone_u, drone_v = map_to_radar(d['x'], d['z'])
            drone_u, drone_v = max(6, min(radar_size - 6, drone_u)), max(6, min(radar_size - 6, drone_v))
            color = (0, 140, 255) if d['gov'] else ((0, 0, 255) if d['local'] else (0, 255, 255))
            cv2.circle(radar_img, (drone_u, drone_v), 6, color, -1)

        return radar_img

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 主界面 GUI
# ==========================================
class DroneMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("综合防区与无人机监控系统 (单目/双目 + 光流测速 + YOLOv11)")
        self.resize(1600, 900)
        self.is_slider_dragging = False 

        self.thread = VideoProcessorThread()
        self.thread.change_pixmap_signal.connect(self.update_images)
        self.thread.finished_signal.connect(self.process_finished)
        self.thread.init_video_signal.connect(self.init_video_player)
        self.thread.update_progress_signal.connect(self.update_progress)
        self.thread.coord_signal.connect(self.update_coord_dashboard)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_widget = QWidget(); left_widget.setFixedWidth(360)
        left_widget.setLayout(left_panel)

        # 视频与视觉模式设置
        file_group = QGroupBox("输入源与视觉模式")
        file_layout = QVBoxLayout()
        self.btn_select_video = QPushButton("导入现场视频")
        self.btn_select_video.clicked.connect(self.select_video)
        file_layout.addWidget(self.btn_select_video)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("视觉模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Mono (单目先验尺寸测距)", "Stereo (双目视差测距)"])
        self.mode_combo.currentTextChanged.connect(self.change_vision_mode)
        mode_layout.addWidget(self.mode_combo)
        file_layout.addLayout(mode_layout)
        
        self.btn_upload_nfz = QPushButton("导入禁飞区 (JSON)")
        self.btn_upload_nfz.clicked.connect(self.upload_nfz_json)
        file_layout.addWidget(self.btn_upload_nfz)
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)

        # 参数输入提取
        self.inputs = {}
        def add_params_group(title, params_list):
            group = QGroupBox(title)
            layout = QFormLayout()
            for key, default, label in params_list:
                le = QLineEdit(default)
                self.inputs[key] = le
                layout.addRow(label + ":", le)
            group.setLayout(layout)
            left_panel.addWidget(group)

        add_params_group("相机绝对地理外参", [
            ('CAM_LAT', '39.9042', '纬度°'), ('CAM_LON', '116.4074', '经度°'), 
            ('CAM_ALT', '50.0', '海拔 (m)'), ('CAM_YAW', '0.0', '偏航角'), ('CAM_PITCH', '-15.0', '俯仰角')
        ])
        
        add_params_group("相机内参与无人机尺寸", [
            ('FX', '1200.0', 'FX'), ('FY', '1200.0', 'FY'), 
            ('CX', '640.0', 'CX'), ('CY', '360.0', 'CY'), 
            ('DRONE_W', '0.4', '目标宽(m)'), ('DRONE_H', '0.15', '目标高(m)'), ('CAMERA_H', '2.0', '相机高度(m)')
        ])

        add_params_group("雷达本场防区 (相对坐标)", [
            ('X_MIN', '-20.0', '左边界'), ('X_MAX', '20.0', '右边界'), 
            ('Z_MIN', '5.0', '近处边界'), ('Z_MAX', '100.0', '远处边界')
        ])

        # 控制按钮
        self.btn_start = QPushButton("启动监测"); self.btn_start.clicked.connect(self.start_process)
        self.btn_stop = QPushButton("停止监测"); self.btn_stop.clicked.connect(self.stop_process)
        left_panel.addWidget(self.btn_start); left_panel.addWidget(self.btn_stop)
        left_panel.addStretch()
        main_layout.addWidget(left_widget)

        # 主显示区域
        display_layout = QHBoxLayout()
        video_area = QVBoxLayout()
        self.video_label = QLabel("等待数据接入...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; color: #fff;")
        video_area.addWidget(self.video_label, stretch=1) 
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        video_area.addWidget(self.slider)
        display_layout.addLayout(video_area, stretch=2)

        # 雷达与仪表盘
        radar_panel = QVBoxLayout()
        self.lbl_dashboard = QLabel("等待目标接入...")
        self.lbl_dashboard.setStyleSheet("background-color: #111; color: #00e676; font-size: 14px; padding: 10px;")
        radar_panel.addWidget(self.lbl_dashboard)

        self.radar_label = QLabel("雷达脱机")
        self.radar_label.setStyleSheet("background-color: #111;")
        self.radar_label.setFixedSize(400, 400) 
        radar_panel.addWidget(self.radar_label)
        
        display_layout.addLayout(radar_panel, stretch=1)
        main_layout.addLayout(display_layout)
        self.setLayout(main_layout)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_name: self.thread.video_path = file_name

    def upload_nfz_json(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "上传禁飞区文件", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                self.thread.gov_nfz_polygon_gps = [(float(pt[0]), float(pt[1])) for pt in json.load(f)]

    def change_vision_mode(self, text):
        self.thread.vision_mode = "Stereo" if "Stereo" in text else "Mono"

    def start_process(self):
        if not self.thread.video_path: return
        for key in self.thread.params.keys():
            if key in self.inputs: self.thread.params[key] = float(self.inputs[key].text())
        self.thread.start()

    def stop_process(self):
        self.thread.stop()

    def init_video_player(self, total_frames, fps):
        self.slider.setRange(0, total_frames)

    def update_progress(self, current_frame, time_str):
        if not self.is_slider_dragging: self.slider.setValue(current_frame)

    def update_coord_dashboard(self, info):
        if info['has_drone']:
            text = (f"【地球绝对位置】\nLat: {info['lat']:.6f}° | Lon: {info['lon']:.6f}°\n"
                    f"绝对海拔(Alt): {info['alt']:.1f}m\n\n"
                    f"【相机相对位置 & 速度】\n"
                    f"横向(X): {info['x']:+.2f}m | 纵深(Z): {info['z']:+.2f}m\n"
                    f"相对高度(Y): {info['y']:+.2f}m\n"
                    f"实时速度(V): {info['speed']:.2f} m/s")
            self.lbl_dashboard.setText(text)
        else:
            self.lbl_dashboard.setText("正在扫描空域...")

    def update_images(self, cv_img, radar_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[2]*rgb_img.shape[1], QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        radar_rgb = cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB)
        qt_radar = QImage(radar_rgb.data, radar_rgb.shape[1], radar_rgb.shape[0], radar_rgb.shape[2]*radar_rgb.shape[1], QImage.Format.Format_RGB888)
        self.radar_label.setPixmap(QPixmap.fromImage(qt_radar).scaled(self.radar_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def process_finished(self):
        self.video_label.setText("监测已结束")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneMonitorApp()
    window.show()
    sys.exit(app.exec())