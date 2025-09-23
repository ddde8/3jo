import cv2
import numpy as np

def detect_lines_in_frame(frame):
    # 1. 이미지를 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거 (블러 처리)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 엣지 감지 (Canny)
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. 허프 변환을 사용하여 직선 감지
    # lines 변수에 시작점과 끝점 좌표가 담깁니다.
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    # 5. 감지된 직선을 원본 이미지에 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 직선을 초록색으로 그립니다.
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    return frame

# 동영상 파일 열기
video_path = "/Users/jieunchoi/Documents/GitHub/3jo/data/IMG_2838-1.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 함수를 호출하여 프레임에서 선을 감지하고 그립니다.
    processed_frame = detect_lines_in_frame(frame)
    
    cv2.imshow("Automated Line Detection", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()