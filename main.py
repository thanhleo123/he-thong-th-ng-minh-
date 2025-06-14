import re
import cv2
import torch
import os
import time
import csv
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import logging
import threading
import queue
from collections import deque, defaultdict
import tkinter as tk
from tkinter import messagebox
import function.utils_rotate as utils_rotate
import function.helper as helper
from modal import IPModal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('process.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedLicensePlateSystem:
    def __init__(self, IP_CAM=None):
        self.IP_CAM = IP_CAM or []
        self.cameras = {}
        self.detection_start_time = None
        self.stats = {
            'total_detected': {},
            'fps_history': {},
            'processing_time': {}
        }
        self.DETECTION_TIME = 1
        self.COOLDOWN_TIME = 6.0
        self.FRAME_SKIP = 3   
        self.MIN_CONFIDENCE = 0.65   
        self.detected_plates_history = defaultdict(list)
        self.PLATE_HISTORY_DURATION = 300
        self.MIN_DETECTION_INTERVAL = 30
        self.WINDOW_NAME = "BAI DO XE THONG MINH"
        self.UI_COLORS = {
            'panel_bg': (30, 30, 30, 180),
            'panel_border': (0, 150, 255),
            'text_primary': (255, 255, 255),
            'highlight': (0, 200, 0),
            'warning': (255, 100, 0),
            'detected_bg': (0, 100, 200, 150)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.pil_font = None
        self.yolo_LP_detect = None
        self.yolo_license_plate = None
        self.is_running = False
        self.processing_threads = {}
        self.frame_queues = {}  
        self.last_detection_time = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    def check_cameras(self):
        available_cameras = []
        if len(self.IP_CAM) < 2:
            logger.error("Less than 2 cameras detected")
            tk.messagebox.showerror("Lỗi", "Cần ít nhất 2 camera được kết nối.")
            exit()
        for idx, ip_url in enumerate(self.IP_CAM):
            cap = cv2.VideoCapture(ip_url)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(ip_url)
                cap.release()
        for idx, ip_url in enumerate(available_cameras):
            cam_type = 'out' if idx % 2 == 0 else 'in'
            self.cameras[idx] = {
                'cap': None,
                'is_active': False,
                'current_detection': None,
                'last_confirmed_plate': None,
                'processing_queue': queue.Queue(maxsize=10),  
                'cam_type': cam_type,
                'ip': ip_url,
                'last_frame': None
            }
            self.stats['total_detected'][idx] = 0
            self.stats['fps_history'][idx] = deque(maxlen=10)
            self.stats['processing_time'][idx] = deque(maxlen=10)
            self.last_detection_time[idx] = 0
            self.frame_queues[idx] = queue.Queue(maxsize=5)  
        return True

    def draw_text_vietnamese(self, frame, text, pos, font_size=16, color=(255, 255, 255)):
        try:
            if not hasattr(self, 'pil_font') or self.pil_font is None:
                font_path = "roboto.ttf"   
                self.pil_font = ImageFont.truetype(font_path, font_size)
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text(pos, text, font=self.pil_font, fill=color[::-1])
            frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(frame, text, pos, self.FONT, 0.6, color, 1, cv2.LINE_AA)

    def is_plate_recently_detected(self, plate_text: str, current_time: float) -> bool:
        recent_detections = [
            t for t in self.detected_plates_history[plate_text]
            if current_time - t < self.PLATE_HISTORY_DURATION
        ]
        self.detected_plates_history[plate_text] = recent_detections
        return any(current_time - t < self.MIN_DETECTION_INTERVAL for t in recent_detections)

    def add_plate_to_history(self, plate_text: str, current_time: float):
        self.detected_plates_history[plate_text].append(current_time)
        self.detected_plates_history[plate_text] = [
            t for t in self.detected_plates_history[plate_text]
            if current_time - t < self.PLATE_HISTORY_DURATION
        ]

    def draw_panel(self, frame, x, y, w, h, bg_color, border_color=None, thickness=2):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        frame = cv2.addWeighted(overlay, bg_color[3]/255.0, frame, 1 - bg_color[3]/255.0, 0)
        if border_color:
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, thickness)
        return frame

    def draw_text(self, frame, text, pos, font_scale=0.6, color=None, bold=False):
        color = color or self.UI_COLORS['text_primary']
        thickness = 2 if bold else 1
        cv2.putText(frame, text, pos, self.FONT, font_scale, color, thickness, cv2.LINE_AA)

    def classify_vehicle_plate(self, plate_number: str) -> str:
        plate_number = plate_number.replace(" ", "").upper()
        patterns = {
            "Xe may dien": [r'^\d{2}MD\d{4,5}$'],
            "Xe may": [
                r'^\d{2}[A-Z]\d-\d{4,5}$',
                r'^\d{2}-[A-Z]\d\d{4,5}$',
                r'^\d{2}-MD-\d{3}\.\d{2}$'
            ],
            "Xe o to": [
                r'^\d{2}[A-Z]{1,2}\d{4,5}$',
                r'^\d{2}[A-Z]{1,2}-\d{4,5}$',
                r'^\d{2}[A-Z]{1,2}-\d{3}\.\d{2}$',
                r'^[A-Z]{2}-\d{3}-\d{2}$'
            ]
        }
        for vehicle_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.match(pattern, plate_number):
                    return vehicle_type
        return "Bien so khong hop le"

    def initialize_camera(self, index: int, ip_url: str) -> bool:
        try:
            cap = cv2.VideoCapture(ip_url)
            if not cap.isOpened():
                logger.error(f"Cannot open camera {index} ::: {ip_url}")
                return False
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 370)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cameras[index]['cap'] = cap
            self.cameras[index]['is_active'] = True
            logger.info(f"Camera {index} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera {index}: {e}")
            return False

    def initialize_models(self) -> bool:
        try:
            if not os.path.exists('model/LP_detector.pt') or not os.path.exists('model/LP_ocr.pt'):
                logger.error("Model files not found")
                return False
            self.yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', source='local').to(self.device)
            self.yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', source='local').to(self.device)
            self.yolo_LP_detect.conf = self.MIN_CONFIDENCE
            self.yolo_license_plate.conf = self.MIN_CONFIDENCE
            self.yolo_LP_detect.eval()
            self.yolo_license_plate.eval()
            return True
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False

    def process_plate_ocr(self, crop_img):
        if crop_img is None or crop_img.size == 0:
            return "unknown"
        try:
            angles = [0, 1, -1]  # Giảm số góc xoay để tối ưu
            for angle in angles:
                rotated = crop_img if angle == 0 else utils_rotate.deskew(crop_img, angle, 0)
                if rotated is not None:
                    lp = helper.read_plate(self.yolo_license_plate, rotated)
                    if lp != "unknown" and len(lp.strip()) >= 6:
                        return lp.strip()
            return "unknown"
        except Exception:
            return "unknown"

    def resize_frame(self, frame, target_width=640, target_height=480):
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas, x_offset, y_offset

    def compose_all_frames(self):
        entry_row = []
        exit_row = []
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for idx, cam in self.cameras.items():
            frame = cam.get('last_frame', black_frame.copy())
            if not cam['is_active']:
                frame = self.draw_camera_not_working(frame, idx)
            if cam['cam_type'] == 'in':
                entry_row.append(frame)
            else:
                exit_row.append(frame)
        entry_row = entry_row or [black_frame.copy()]
        exit_row = exit_row or [black_frame.copy()]
        return np.vstack([np.hstack(entry_row), np.hstack(exit_row)])

    def draw_modern_ui(self, frame, cam_index):
        h, w = frame.shape[:2]
        title = "CAMERA VAO" if self.cameras[cam_index]['cam_type'] == 'in' else "CAMERA RA"
        frame = self.draw_panel(frame, 0, 0, w, 50, self.UI_COLORS['panel_bg'], self.UI_COLORS['panel_border'])
        self.draw_text_vietnamese(frame, title, (20, 15), font_size=20, color=self.UI_COLORS['panel_border'])
        return frame

    def draw_detection_box(self, frame, cam_index, x_offset=0, y_offset=0):
        if not self.cameras[cam_index]['current_detection']:
            return frame
        x, y, x2, y2 = self.cameras[cam_index]['current_detection']['bbox']
        x, y, x2, y2 = x + x_offset, y + y_offset, x2 + x_offset, y2 + y_offset
        color = self.UI_COLORS['warning'] if self.cameras[cam_index]['current_detection'].get('spam_warning') else self.UI_COLORS['highlight']
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        return frame

    def draw_confirmed_plate(self, frame, cam_index):
        if not self.cameras[cam_index]['last_confirmed_plate']:
            return frame
        h, w = frame.shape[:2]
        panel_w, panel_h = 300, 180
        x, y = w - panel_w - 20, 60
        frame = self.draw_panel(frame, x, y, panel_w, panel_h, self.UI_COLORS['detected_bg'], self.UI_COLORS['highlight'])
        try:
            crop = self.cameras[cam_index]['last_confirmed_plate']['crop']
            cimg = cv2.resize(crop, (panel_w-20, 90), interpolation=cv2.INTER_LINEAR)
            frame[y+10:y+100, x+10:x+panel_w-10] = cimg
        except:
            pass
        plate = self.cameras[cam_index]['last_confirmed_plate']['plate']
        v_type = self.cameras[cam_index]['last_confirmed_plate']['vehicle_type']
        ts = self.cameras[cam_index]['last_confirmed_plate']['timestamp'][:19]
        self.draw_text_vietnamese(frame, ts, (x+10, y+130), font_size=14)
        self.draw_text_vietnamese(frame, f"Bien so: {plate}", (x+10, y+150), font_size=16)
        self.draw_text_vietnamese(frame, f"Loai: {v_type}", (x+10, y+170), font_size=14)
        return frame

    def draw_camera_not_working(self, frame, cam_index):
        h, w = frame.shape[:2]
        title = "CAMERA VAO" if self.cameras[cam_index]['cam_type'] == 'in' else "CAMERA RA"
        self.draw_text_vietnamese(frame, f"{title} khong hoat dong", (w//2-150, h//2), font_size=20, color=self.UI_COLORS['warning'])
        return frame

    def frame_reader(self, idx):
        cap = self.cameras[idx]['cap']
        while self.is_running and self.cameras[idx]['is_active']:
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    self.frame_queues[idx].put_nowait(frame)
                except queue.Full:
                    pass
            time.sleep(0.001)

    def processing_worker(self, cam_index):
        prev_time = time.time()
        frame_count = 0
        while self.is_running:
            try:
                frame = self.frame_queues[cam_index].get(timeout=0.1)
                frame_count += 1
                now = time.time()
                if frame_count % self.FRAME_SKIP == 0:
                    frame, x_offset, y_offset = self.resize_frame(frame, 640, 480)
                    self.cameras[cam_index]['processing_queue'].put_nowait((frame.copy(), now))
                fps = frame_count / (now - prev_time + 1e-6)
                self.stats['fps_history'][cam_index].append(fps)
                frame, x_offset, y_offset = self.resize_frame(frame, 640, 480)
                frame = self.draw_modern_ui(frame, cam_index)
                frame = self.draw_detection_box(frame, cam_index, x_offset, y_offset)
                frame = self.draw_confirmed_plate(frame, cam_index)
                avg_fps = sum(self.stats['fps_history'][cam_index]) / len(self.stats['fps_history'][cam_index]) if self.stats['fps_history'][cam_index] else 0
                self.draw_text_vietnamese(frame, f"FPS: {int(avg_fps)}", (20, 40), font_size=14)
                self.cameras[cam_index]['last_frame'] = frame
                if now - prev_time > 1:
                    prev_time = now
                    frame_count = 0
            except queue.Empty:
                continue
            except queue.Full:
                pass

    def detection_worker(self, cam_index):
        while self.is_running:
            try:
                frame_data = self.cameras[cam_index]['processing_queue'].get(timeout=0.1)
                frame, current_time = frame_data
                self.process_frame(frame, current_time, cam_index)
                self.cameras[cam_index]['processing_queue'].task_done()
            except queue.Empty:
                continue

    def process_frame(self, frame, current_time, cam_index):
        start_time = time.time()
        with torch.no_grad():  
            plates = self.yolo_LP_detect(frame, size=320)   
        list_plates = plates.pandas().xyxy[0].values.tolist()
        
        if not list_plates:
            if self.cameras[cam_index]['current_detection']:
                self.cameras[cam_index]['current_detection'] = None
            return
        
        best_plate = max(list_plates, key=lambda x: x[4])  
        confidence = best_plate[4]
        
        if confidence < self.MIN_CONFIDENCE:
            return
        
        x, y, xmax, ymax = [int(coord) for coord in best_plate[:4]]
        w, h = xmax - x, ymax - y
        if w < 60 or h < 20:   
            return
        
        crop_img = frame[y:y+h, x:x+w]
        if crop_img.size == 0:
            return
        
        plate_text = self.process_plate_ocr(crop_img)
        if plate_text == "unknown" or len(plate_text) < 6:
            return
        
        vehicle_type = self.classify_vehicle_plate(plate_text)
        is_spam = self.is_plate_recently_detected(plate_text, current_time)
        
        self.cameras[cam_index]['current_detection'] = {
            'plate': plate_text,
            'vehicle_type': vehicle_type,
            'crop': crop_img.copy(),
            'bbox': (x, y, xmax, ymax),
            'confidence': confidence,
            'spam_warning': is_spam
        }
        
        HIGH_CONFIDENCE_THRESHOLD = 0.85   
        if confidence >= HIGH_CONFIDENCE_THRESHOLD and not is_spam and \
        (not self.cameras[cam_index]['last_confirmed_plate'] or 
            self.cameras[cam_index]['last_confirmed_plate']['plate'] != plate_text or
            current_time - self.last_detection_time[cam_index] > self.COOLDOWN_TIME):
            self.confirm_detection(current_time, frame, cam_index)
        
        self.stats['processing_time'][cam_index].append((time.time() - start_time) * 100)

    def confirm_detection(self, current_time, frame, cam_index):
        if not self.cameras[cam_index]['current_detection']:
            return
        plate_text = self.cameras[cam_index]['current_detection']['plate']
        vehicle_type = self.cameras[cam_index]['current_detection']['vehicle_type']
        crop_img = self.cameras[cam_index]['current_detection']['crop']
        confidence = self.cameras[cam_index]['current_detection']['confidence']
        self.add_plate_to_history(plate_text, current_time)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cameras[cam_index]['last_confirmed_plate'] = {
            'plate': plate_text,
            'vehicle_type': vehicle_type,
            'crop': crop_img.copy(),
            'timestamp': timestamp,
            'bbox': self.cameras[cam_index]['current_detection']['bbox']
        }
        self.save_detection_data(timestamp, frame, cam_index)
        self.stats['total_detected'][cam_index] += 1
        self.last_detection_time[cam_index] = current_time
        logger.info(f"Camera {cam_index} CONFIRMED: {plate_text} ({vehicle_type}) with confidence {confidence:.2f}")
        self.cameras[cam_index]['current_detection'] = None

    def save_detection_data(self, timestamp, frame, cam_index):
        try:
            if not self.cameras[cam_index]['last_confirmed_plate']:
                return
            ts_file = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
            image_path = f"output/images/cam{cam_index}_img_{ts_file}.jpg"
            crop_path = f"output/crops/cam{cam_index}_crop_{ts_file}.jpg"
            cv2.imwrite(image_path, frame)
            cv2.imwrite(crop_path, self.cameras[cam_index]['last_confirmed_plate']['crop'])
            self.save_to_csv(timestamp, image_path, crop_path, cam_index)
        except Exception as e:
            logger.error(f"Data saving error for camera {cam_index}: {e}")

    def save_to_csv(self, timestamp, image_path, crop_path, cam_index):
        try:
            csv_file = "output/data/detections.csv"
            plate_data = self.cameras[cam_index]['last_confirmed_plate']
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    plate_data['plate'],
                    plate_data['vehicle_type'],
                    str(plate_data['bbox']),
                    image_path,
                    crop_path,
                    self.cameras[cam_index]['cam_type']
                ])
        except Exception as e:
            logger.error(f"CSV saving error for camera {cam_index}: {e}")

    def setup_directories(self):
        for d in ["output/images", "output/crops", "output/data"]:
            os.makedirs(d, exist_ok=True)
        csv_file = "output/data/detections.csv"
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'plate', 'vehicle_type', 'bbox', 'image_path', 'crop_path', 'cam_type'])

    def run(self):
        self.setup_directories()
        if not self.check_cameras():
            logger.error("Không tìm thấy camera nào.")
            self.cleanup()
            return

        # Khởi tạo camera
        for idx in self.cameras:
            if not self.initialize_camera(idx, self.cameras[idx]['ip']):
                logger.warning(f"Camera {idx} không mở được")

        # Khởi tạo model
        if not self.initialize_models():
            logger.error("Lỗi không tải được model")
            self.cleanup()
            return

        # Kiểm tra camera hoạt động
        if not any(self.cameras[idx]['is_active'] for idx in self.cameras):
            messagebox.showerror("Lỗi", "Không có camera nào hoạt động.")
            self.cleanup()
            return

        self.is_running = True

        # Khởi động các luồng
        for idx in self.cameras:
            if self.cameras[idx]['is_active']:
                threading.Thread(target=self.frame_reader, args=(idx,), daemon=True).start()
                threading.Thread(target=self.processing_worker, args=(idx,), daemon=True).start()
                threading.Thread(target=self.detection_worker, args=(idx,), daemon=True).start()

        # Tạo cửa sổ OpenCV
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 640, 640)

        try:
            while self.is_running:
                try:
                    # Lấy frame kết hợp từ compose_all_frames
                    combined_frame = self.compose_all_frames()

                    # Kiểm tra frame hợp lệ
                    if combined_frame is None or not isinstance(combined_frame, np.ndarray):
                        logger.warning("Frame không hợp lệ, bỏ qua hiển thị.")
                        time.sleep(0.01)  # Tránh vòng lặp quá nhanh khi lỗi
                        continue

                    # Đảm bảo frame là RGB (3 kênh) để hiển thị
                    if len(combined_frame.shape) == 2:  # Nếu là grayscale
                        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2RGB)
                    elif len(combined_frame.shape) != 3 or combined_frame.shape[2] != 3:
                        logger.error(f"Frame có shape không hợp lệ: {combined_frame.shape}")
                        continue

                    # Hiển thị frame
                    cv2.imshow(self.WINDOW_NAME, combined_frame)

                    # Xử lý phím bấm
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Người dùng yêu cầu thoát.")
                        break
                    elif key == ord('r'):
                        for idx in self.cameras:
                            self.cameras[idx]['current_detection'] = None
                            self.cameras[idx]['last_confirmed_plate'] = None
                            self.stats['total_detected'][idx] = 0
                        logger.info("Khởi động lại hệ thống.")
                    elif key == ord('c'):
                        self.detected_plates_history.clear()
                        logger.info("Xóa dữ liệu lịch sử.")

                except Exception as e:
                    logger.error(f"Lỗi trong vòng lặp chính: {str(e)}")
                    time.sleep(0.01)  # Tránh vòng lặp quá nhanh khi lỗi

        except KeyboardInterrupt:
            logger.info("Đã nhận tín hiệu ngắt từ người dùng.")

        finally:
            self.is_running = False
            self.cleanup()
            cv2.destroyAllWindows()
            logger.info("Hệ thống đã tắt.")

    def cleanup(self):
        self.is_running = False
        for idx in self.cameras:
            if self.cameras[idx].get('cap'):
                self.cameras[idx]['cap'].release()
        cv2.destroyAllWindows()
        logger.info("Cleanup xong!")

def main():
    try:
        root = tk.Tk()
        root.title("Khởi động hệ thống")
        root.geometry("300x150")
        root.resizable(False, False)
        system = OptimizedLicensePlateSystem()
        def start_system():
            modal = IPModal(root)
            ips = modal.get_ips()
            if ips:
                system.IP_CAM = ips
                root.quit()
                root.destroy()
                system.run()
        start_button = tk.Button(
            root, text="Bắt đầu hệ thống", command=start_system,
            font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10
        )
        start_button.pack(expand=True)
        label = tk.Label(root, text="Nhấn nút để nhập IP camera và khởi động", font=("Arial", 10))
        label.pack(pady=(0, 20))
        root.mainloop()
    except Exception as e:
        logger.error(f"Hệ thống lỗi: {e}")

if __name__ == "__main__":
    main()