import os
import json
import cv2
import threading
import time
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
# from ultralytics import YOLO # YOLO 사용 시 주석 해제

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 전역 변수 (서버 상태 관리)
PARKING_SPOTS_DEFINED = False
VIDEO_PATH = None
PARKING_SPOTS = {} # {'P1': {'coords': [...], 'status': 'available', 'reserved_until': None, ...}}
RESERVATION_HOLD_TIME = 300 # 5분 (단위: 초)

# --------------------
# 💡 주차장 분석 및 예약 모니터링 로직
# --------------------
def reservation_monitor():
    """
    백그라운드에서 예약 시간을 감시하고, 만료된 예약을 취소합니다.
    """
    while True:
        now = time.time()
        for spot_id in list(PARKING_SPOTS.keys()):
            spot = PARKING_SPOTS[spot_id]
            if spot.get("status") == "reserved" and now > spot.get("reserved_until"):
                print(f"예약 시간 만료: {spot_id} 예약 취소")
                spot["status"] = "available"
                spot["reserved_until"] = None
        time.sleep(1) # 1초마다 확인

def analyze_video():
    """
    YOLO로 동영상 분석 및 주차장 상태를 업데이트하는 함수.
    """
    global PARKING_SPOTS
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # YOLO 모델 로드 (주석 해제 후 사용)
    # model = YOLO('yolov8n.pt') 

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # ----------------------------------------------------
        # 💡 실제 YOLO 감지 및 주차 공간 점유 확인 로직
        # ----------------------------------------------------
        # 지금은 테스트를 위해 임의의 데이터로 대체합니다.
        import random
        for spot_id in PARKING_SPOTS:
            is_occupied_in_frame = random.choice([True, False])
            
            spot = PARKING_SPOTS[spot_id]
            
            # CASE 1: 차량이 감지되었을 때
            if is_occupied_in_frame:
                if spot.get("status") in ["available", "reserved"]:
                    spot["status"] = "occupied"
                    spot["occupied_since"] = time.time()
                    spot["reserved_until"] = None # 예약 취소
            
            # CASE 2: 차량이 감지되지 않았을 때
            elif spot.get("status") == "occupied":
                # 비점유 상태로 변경
                spot["status"] = "available"
                spot["occupied_since"] = None

        time.sleep(1) 

def generate_video_feed():
    """
    웹으로 실시간 비디오 프레임을 스트리밍하는 함수.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        for spot_id, spot_data in PARKING_SPOTS.items():
            if "coords" in spot_data:
                x1, y1, x2, y2 = spot_data["coords"]
                
                # 상태에 따라 색상 결정
                status = spot_data.get("status")
                if status == "occupied":
                    color = (0, 0, 255) # 빨강
                elif status == "reserved":
                    color = (255, 165, 0) # 주황색
                else:
                    color = (0, 255, 0) # 초록색
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --------------------
# 💡 Flask 라우트 (API 엔드포인트)
# --------------------
@app.route('/')
def index():
    if PARKING_SPOTS_DEFINED:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/set_parking_data', methods=['POST'])
def set_parking_data():
    global PARKING_SPOTS_DEFINED, VIDEO_PATH, PARKING_SPOTS
    
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
    주차 공간을 예약하는 API 엔드포인트
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

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/parking_status')
def get_parking_status():
    status_copy = PARKING_SPOTS.copy()
    # reserved_until 값을 남은 시간으로 변환
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