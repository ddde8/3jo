from sprite import Sprite
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class TextSprite(Sprite):
    """텍스트 스프라이트 클래스"""
    def __init__(self, x, y, text, font_size=30, color=(255, 0, 0, 0), font_path='data/NanumPenScript-Regular.ttf'):
        super().__init__(x, y)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.font_path = font_path
        self._create_text_image()

    def _create_text_image(self):
        """한글 텍스트 이미지 생성"""
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except:
            font = ImageFont.load_default()

        # 임시 이미지로 텍스트 크기 측정
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), self.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 텍스트 크기에 맞는 이미지 생성 (여백 추가)
        margin = 5
        img_width = text_width + margin * 2
        img_height = text_height + margin * 2

        pil_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        # 여백을 고려한 텍스트 위치 (bbox의 음수 오프셋 보정)
        x = margin - bbox[0]
        y = margin - bbox[1]

        draw.text((x, y), self.text, font=font, fill=self.color)

        # RGB로 변환 후 OpenCV 이미지로 변환
        pil_img = pil_img.convert('RGB')
        self.image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.width = img_width
        self.height = img_height

    def set_text(self, new_text):
        """텍스트 변경"""
        self.text = new_text
        self._create_text_image()

    def set_color(self, new_color):
        """색상 변경"""
        self.color = new_color
        self._create_text_image()
    
    def update(self):
        self.color = ((self.color[0]+7)%256, (self.color[1]+3)%256, (self.color[2]+1)%256, 0)
        self.set_color(self.color)
