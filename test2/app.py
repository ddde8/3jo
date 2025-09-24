import os
import json
import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for server state management
PARKING_SPOTS_DEFINED = False
VIDEO_PATH = None
PARKING_SPOTS = {} # {'P1': {'coords': [...], 'status': 'available', 'reserved_until': None, ...}}
RESERVATION_HOLD_TIME = 300 # 5분 (단위: 초)

# Shared variables for background threads
latest_frame = None
latest_frame_lock = threading.Lock()
latest_yolo_frame = None
latest_yolo_frame_lock = threading.Lock()

# YOLO-Seg threshold and delay settings
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3 # 겹침 비율 임계값
OCCUPIED_RELEASE_DELAY = 3  # 차량이 사라진 후 3초 동안 점유 상태 유지

# --------------------
# 💡 Helper functions
# --------------------
def create_mask_from_coords(coords, frame_shape):
    """
    주어진 좌표를 바탕으로 이진 마스크를 생성합니다.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.array([[coords[0], coords[1]], [coords[2], coords[1]], [coords[2], coords[3]], [coords[0], coords[3]]], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def calculate_iou_from_masks(mask1, mask2):
    """
    두 마스크의 IoU(Intersection over Union)를 계산합니다.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

# --------------------
# 💡 Parking Analysis and Reservation Monitoring Logic
# --------------------
def reservation_monitor():
    """
    Monitors reservation times in the background and cancels expired reservations.
    """
    while True:
        now = time.time()
        for spot_id in list(PARKING_SPOTS.keys()):
            spot = PARKING_SPOTS[spot_id]
            if spot.get("status") == "reserved" and now > spot.get("reserved_until", 0):
                print(f"예약 시간 만료: {spot_id} 예약 취소")
                spot["status"] = "available"
                spot["reserved_until"] = None
        time.sleep(1)

def analyze_video():
    """
    Loads the YOLO-Seg model and continuously analyzes video frames for car segmentation.
    Updates the parking spot status and shares the latest processed frame.
    """
    global PARKING_SPOTS, latest_frame, latest_yolo_frame

    if VIDEO_PATH is None:
        print("Error: Video path is not set.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}.")
        return

    try:
        # Load the YOLO-Seg model
        model = YOLO('yolov8n-seg.pt')
        print(f"YOLO-Seg model loaded. Class names: {model.names}")
    except Exception as e:
        print(f"Error loading YOLO-Seg model: {e}")
        return

    # 첫 프레임을 읽어와서 주차 공간 마스크를 미리 생성합니다.
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return
    
    parking_spot_masks = {}
    for spot_id, spot in PARKING_SPOTS.items():
        parking_spot_masks[spot_id] = create_mask_from_coords(spot["coords"], initial_frame.shape)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 비디오를 다시 처음으로 돌립니다.

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        results = model(frame, verbose=False)[0]
        
        detected_car_masks = []
        if results.masks is not None:
            for mask, cls, conf in zip(results.masks.data.cpu().numpy(),
                                      results.boxes.cls.cpu().numpy(),
                                      results.boxes.conf.cpu().numpy()):
                if int(cls) in [2, 5, 7] and conf > CONF_THRESHOLD:
                    detected_car_masks.append(mask)

        # Draw YOLO segmentation masks on a separate frame for the left panel
        yolo_frame = frame.copy()
        if results.masks is not None:
            # 마스크를 오버레이
            for i, mask in enumerate(results.masks.data):
                yolo_frame = results.plot(conf=False, labels=False, boxes=False, masks=True)
                yolo_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2RGB)
                break # 첫 번째 객체 마스크만 플로팅
                
        with latest_yolo_frame_lock:
            latest_yolo_frame = yolo_frame.copy()

        # Update parking spot status based on detected car masks
        for spot_id, spot in PARKING_SPOTS.items():
            is_occupied = False
            spot_mask = parking_spot_masks[spot_id]
            
            for car_mask in detected_car_masks:
                # 차량 마스크와 주차 공간 마스크의 겹침 비율 계산
                iou_score = calculate_iou_from_masks(spot_mask, car_mask)
                if iou_score > IOU_THRESHOLD:
                    is_occupied = True
                    break
            
            if is_occupied:
                if spot.get("status") in ["available", "reserved"]:
                    spot["status"] = "occupied"
                    spot["occupied_since"] = time.time()
                    spot["reserved_until"] = None
            else:
                if spot.get("status") == "occupied":
                    occupied_since = spot.get("occupied_since") or 0
                    if time.time() - occupied_since > OCCUPIED_RELEASE_DELAY:
                        spot["status"] = "available"
                        spot["occupied_since"] = None

        with latest_frame_lock:
            latest_frame = frame.copy()

        time.sleep(1)

def generate_yolo_feed():
    """
    Streams the live video feed with only YOLO-Seg detected objects.
    """
    while True:
        with latest_yolo_frame_lock:
            if latest_yolo_frame is None:
                continue
            frame = latest_yolo_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_video_feed():
    """
    Streams the live video feed with parking spot status overlaid.
    """
    while True:
        with latest_frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        if PARKING_SPOTS_DEFINED and PARKING_SPOTS:
            for spot_id, spot_data in PARKING_SPOTS.items():
                if "coords" in spot_data and isinstance(spot_data["coords"], list) and len(spot_data["coords"]) == 4:
                    x1, y1, x2, y2 = spot_data["coords"]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    status = spot_data.get("status")
                    if status == "occupied":
                        color = (0, 0, 255) # Red
                    elif status == "reserved":
                        color = (255, 165, 0) # Orange
                    else:
                        color = (0, 255, 0) # Green
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --------------------
# 💡 Flask Routes (API Endpoints)
# --------------------
@app.route('/')
def index():
    if PARKING_SPOTS_DEFINED:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/set_parking_data', methods=['POST'])
def set_parking_data():
    global PARKING_SPOTS_DEFINED, VIDEO_PATH, PARKING_SPOTS, latest_frame
    
    video_file = request.files.get('video')
    lines_json = request.form.get('lines')
    
    if not video_file or not lines_json:
        return jsonify({"message": "파일 또는 선 정보가 누락되었습니다."}), 400

    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        os.remove(VIDEO_PATH)

    filename = secure_filename(video_file.filename)
    VIDEO_PATH = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(VIDEO_PATH)
    
    all_lines = json.loads(lines_json)
    PARKING_SPOTS = {}
    for i, line in enumerate(all_lines):
        PARKING_SPOTS[f"P{i+1}"] = {
            "coords": [min(line['x1'], line['x2']), min(line['y1'], line['y2']),
                       max(line['x1'], line['x2']), max(line['y1'], line['y2'])],
            "status": "available",
            "occupied_since": None,
            "reserved_until": None
        }

    PARKING_SPOTS_DEFINED = True
    
    threading.Thread(target=analyze_video, daemon=True).start()
    threading.Thread(target=reservation_monitor, daemon=True).start()
    
    return jsonify({"message": "데이터가 성공적으로 설정되었습니다."})

@app.route('/reserve/<spot_id>', methods=['POST'])
def reserve_spot(spot_id):
    """
    API endpoint to reserve a parking spot.
    """
    global PARKING_SPOTS
    spot = PARKING_SPOTS.get(spot_id)
    if spot and spot.get("status") == "available":
        spot["status"] = "reserved"
        spot["reserved_until"] = time.time() + RESERVATION_HOLD_TIME
        return jsonify({"message": f"{spot_id}가 성공적으로 예약되었습니다.", "success": True})
    return jsonify({"message": "예약할 수 없는 공간입니다.", "success": False})

@app.route('/dashboard')
def dashboard():
    if not PARKING_SPOTS_DEFINED:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/yolo_feed')
def yolo_feed():
    return Response(generate_yolo_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/parking_status')
def get_parking_status():
    status_copy = PARKING_SPOTS.copy()
    for spot_id, spot_data in status_copy.items():
        if spot_data.get("status") == "reserved" and spot_data.get("reserved_until"):
            remaining_time = int(spot_data["reserved_until"] - time.time())
            if remaining_time > 0:
                spot_data["remaining_time"] = remaining_time
            else:
                spot_data["remaining_time"] = 0
    return jsonify(status_copy)

if __name__ == '__main__':
    app.run(debug=True)
