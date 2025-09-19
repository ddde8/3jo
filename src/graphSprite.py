import cv2
from sprite import Sprite
import numpy as np
from textSprite import TextSprite

class GraphSprite(Sprite):
    """통계(그래프) 스프라이트"""
    def __init__(self, x, y, size=(640, 480)):
        super().__init__(x, y)
        self.size = size
        self.image = self._create_graph_image()

    def _create_graph_image(self):
        # matplotlib 없이 임의의 그래프(막대그래프) 이미지를 생성
        img = np.ones((self.size[1], self.size[0], 3), np.uint8) * 255
        values = [100, 200, 150, 300, 250]
        bar_width = 80
        for i, v in enumerate(values):
            x1 = 50 + i * (bar_width + 20)
            y1 = self.size[1] - 30
            x2 = x1 + bar_width
            y2 = y1 - v
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255), -1)
            cv2.putText(img, str(v), (x1+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        # 한글 텍스트를 TextSprite로 생성하여 draw로 합성
        text_sprite = TextSprite(0, 0, "통계 그래프 예시", font_size=36, color=(255,0,0))
        # 중앙 상단에 배치
        h, w = text_sprite.image.shape[:2]
        x_offset = self.size[0]//2 - w//2
        y_offset = 20
        text_sprite.x = x_offset
        text_sprite.y = y_offset
        text_sprite.draw(img)
        return img

    def update(self):
        pass

    def draw(self, target_img):
        if self.image is not None:
            h, w = self.image.shape[:2]
            target_img[self.y:self.y+h, self.x:self.x+w] = self.image
            target_img[self.y:self.y+h, self.x:self.x+w] = self.image
