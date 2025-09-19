import cv2
import numpy as np
from sprite import Sprite
from textSprite import TextSprite

class ButtonSprite(Sprite):
    """화면 전환 버튼"""
    def __init__(self, x, y, width=150, height=50, text="화면 전환", font_scale=1, text_color=(255,0,0)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.text = text
        self.font_scale = font_scale
        self.text_color = text_color
        self.mode = 0
        self.mode_texts = [text]
        self._create_button_image()

    def _create_button_image(self):
        # 한글 지원을 위해 TextSprite 사용
        self.text_sprite = TextSprite(0, 0, self.text, font_size=int(self.font_scale * 30), color=self.text_color)
        self.image = self.text_sprite.image
        self.image = cv2.resize(self.image, (self.width, self.height))
        cv2.rectangle(self.image, (0, 0), (self.width-1, self.height-1), (255, 0, 0), 2)

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def check_mouse_position(self, mx, my):
        return self.x <= mx < self.x + self.width and self.y <= my < self.y + self.height

    def click(self):
        self.mode += 1
        self.mode = self.mode % len(self.mode_texts)
        self.text = self.mode_texts[self.mode]
        self._create_button_image()
        return self.mode

    def update(self):
        pass
