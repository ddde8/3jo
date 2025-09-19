import cv2
import numpy as np
from videoSprite import VideoSprite
from graphSprite import GraphSprite

# graphSprite는 아래에서 정의

def main():
    screen_w, screen_h = 900, 600
    canvas = np.ones((screen_h, screen_w, 3), np.uint8) * 220

    # 스프라이트 생성
    video_sprite = VideoSprite(130, 100, video_source=0, size=(640, 480))
    graph_sprite = GraphSprite(130, 100, size=(640, 480))
    button_sprite = ButtonSprite(370, 20, width=160, height=60, text="화면 전환")

    # 상태: 0=영상, 1=그래프
    state = [0]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_sprite.check_mouse_position(x, y):
                state[0] = 1 - state[0]  # 화면 전환

    cv2.namedWindow("main")
    cv2.setMouseCallback("main", mouse_callback)

    while True:
        frame = canvas.copy()
        button_sprite.draw(frame)
        if state[0] == 0:
            video_sprite.update()
            video_sprite.draw(frame)
        else:
            graph_sprite.draw(frame)
        cv2.imshow("main", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    if video_sprite.cap is not None:
        video_sprite.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
