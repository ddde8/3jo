import cv2
import numpy as np
from sprite import Sprite

class VideoSprite(Sprite):
    def __init__(self, x, y, video_source=0, size=(640, 480)):
        super().__init__(x, y)
        self.size = size
        self.cap = cv2.VideoCapture(video_source)
        self.image = np.zeros((size[1], size[0], 3), np.uint8)

    def update(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.size)
                self.image = frame
            else:
                self.image = np.zeros_like(self.image)

    def draw(self, target_img):
        if self.image is not None:
            h, w = self.image.shape[:2]
            target_img[self.y:self.y+h, self.x:self.x+w] = self.image
