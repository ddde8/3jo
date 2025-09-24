import os
import json
import cv2
import threading
import time
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
latest_detections = []
latest_detections_lock = threading.Lock()

# 신뢰도 임계값 설정
CONF_THRESHOLD = 0.5
OCCUPIED_RELEASE_DELAY = 3  # 차량 사라진 후 3초 유지
PORT = 5001

# --------------------
# 💡 Helper functions
# --------------------
def rect_overlap(rect1, rect2):
    """
    두 사각형이 조금이라도 겹치면 True를 반환합니다.
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    horizontal_overlap = (x1_min < x2_max) and (x2_min < x1_max)
    vertical_overlap = (y1_min < y2_max) and (y2_min < y1_max)
    
    return horizontal_overlap and vertical_overlap

# --------------------
# 💡 Parking Analysis and Reservation Monitoring Logic
# --------------------
def reservation_monitor():
    """
    백그라운드에서 예약 시간을 감시하고, 만료된 예약을 취소합니다.
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
    YOLO 모델을 로드하고 비디오 프레임을 분석하여 주차 공간 상태를 업데이트합니다.
    """
    global PARKING_SPOTS, latest_frame, latest_detections

    if VIDEO_PATH is None:
        print("Error: Video path is not set.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}.")
        return

    try:
        # Load the YOLO model just once, outside the loop
        model = YOLO('yolo11n.pt')
        print(f"YOLO model loaded. Class names: {model.names}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Perform object detection on the current frame
        results = model(frame, verbose=False)[0]
        
        detected_cars = []
        all_detections = []
        # Iterate through all detected objects with confidence filtering
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy()):
            class_name = model.names[int(cls)]
            detection_info = {
                'box': box.tolist(),
                'class': class_name,
                'confidence': float(conf)
            }
            all_detections.append(detection_info)
            
            # Filter for cars, buses, and trucks (COCO dataset class IDs: car=2, bus=5, truck=7)
            if int(cls) in [2, 5, 7] and conf > CONF_THRESHOLD:
                detected_cars.append(box)

        # Update parking spot status based on detected cars
        for spot_id, spot in PARKING_SPOTS.items():
            is_occupied = False
            spot_rect = spot["coords"]

            for car_box in detected_cars:
                # 겹치는 부분이 조금이라도 있으면 True를 반환
                if rect_overlap(spot_rect, car_box):
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

        # Store the processed frame and detections in the shared variables
        with latest_frame_lock:
            latest_frame = frame.copy()
        
        with latest_detections_lock:
            latest_detections = all_detections.copy()

        # Pause to prevent high CPU usage
        time.sleep(1)

def generate_video_feed():
    """
    Streams the live video feed with parking spot status overlaid.
    """
    while True:
        with latest_frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Overlay parking spot status on the frame
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

def generate_analysis_feed():
    """
    YOLO 분석 결과와 바운딩 박스를 보여주는 비디오 피드를 스트리밍합니다.
    """
    while True:
        with latest_frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        with latest_detections_lock:
            detections = latest_detections.copy()

        # 모든 YOLO 탐지 결과에 바운딩 박스와 레이블 그리기
        for detection in detections:
            box = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, box)
            
            # 클래스에 따른 색상 설정
            if class_name in ['car', 'truck', 'bus']:
                color = (0, 255, 255)  # 노란색 - 차량
            elif class_name == 'person':
                color = (255, 0, 0)    # 파란색 - 사람
            else:
                color = (128, 128, 128) # 회색 - 기타
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블과 신뢰도 표시
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 레이블 배경 그리기
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 레이블 텍스트 그리기
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 주차 공간도 함께 표시
        if PARKING_SPOTS_DEFINED and PARKING_SPOTS:
            for spot_id, spot_data in PARKING_SPOTS.items():
                if "coords" in spot_data and isinstance(spot_data["coords"], list) and len(spot_data["coords"]) == 4:
                    x1, y1, x2, y2 = spot_data["coords"]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    status = spot_data.get("status")
                    if status == "occupied":
                        color = (0, 0, 255) # 빨간색
                        thickness = 3
                    elif status == "reserved":
                        color = (0, 165, 255) # 주황색
                        thickness = 3
                    else:
                        color = (0, 255, 0) # 초록색
                        thickness = 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # 주차 공간 ID 표시
                    cv2.putText(frame, spot_id, (x1 + 5, y1 + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 통계 정보 표시
        stats_text = f"Total Objects: {len(detections)}"
        vehicles = len([d for d in detections if d['class'] in ['car', 'truck', 'bus']])
        vehicles_text = f"Vehicles: {vehicles}"
        
        cv2.putText(frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, vehicles_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --------------------
# 💡 Flask Routes (API Endpoints)
# --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parking_setup')
def parking_setup():
    return render_template('parking_setup.html')

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
    
    # Start the background threads for analysis and reservation monitoring
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

@app.route('/analysis')
def analysis():
    if not PARKING_SPOTS_DEFINED:
        return redirect(url_for('index'))
    return render_template('analysis.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analysis_feed')
def analysis_feed():
    return Response(generate_analysis_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    app.run(debug=True, host='0.0.0.0', port=PORT)
