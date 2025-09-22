import cv2
import numpy as np
from sprite import Sprite
from ultralytics import YOLO

class VideoSprite(Sprite):
    def __init__(self, x, y, video_source=0, size=(640, 480)):
        super().__init__(x, y)
        self.size = size
        self.cap = cv2.VideoCapture(video_source)
        # self.cap = cv2.VideoCapture('data/video2.mp4')
        self.image = np.zeros((size[1], size[0], 3), np.uint8)
        self.model = YOLO('yolo11n.pt', verbose=False)

    def update(self):
        if self.cap is not None:
            # ret, frame = self.cap.read()
            frame = cv2.imread('data/video3.jpg')
            if frame is None:
                ret = False
            else:
                ret = True
            if ret:
                frame = cv2.resize(frame, self.size)
                frame = self.model.predict(frame, conf=0.5, save=False, save_txt=False, line_thickness=2)[0].plot()
                self.image = frame
            else:
                self.image = np.zeros_like(self.image)

    def draw(self, target_img):
        if self.image is not None:
            h, w = self.image.shape[:2]
            target_img[self.y:self.y+h, self.x:self.x+w] = self.image
