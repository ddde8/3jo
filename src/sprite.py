import cv2

class Sprite:
    """기본 스프라이트 클래스"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = None
        self.width = 0
        self.height = 0

    def update(self):
        """스프라이트 업데이트 (오버라이드 필요)"""
        pass

    def draw(self, target_img):
        """스프라이트를 타겟 이미지에 그리기"""
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def _blit(self, img, x, y, src_img, mask=None):
        """이미지에 다른 이미지를 블릿"""
        h, w = src_img.shape[:2]
        if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
            return img

        roi = img[y:y+h, x:x+w]

        # 소스 이미지의 검은색 배경을 마스크로 사용
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        if mask is None:
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        src_fg = cv2.bitwise_and(src_img, src_img, mask=mask)

        dst = cv2.add(img_bg, src_fg)
        img[y:y+h, x:x+w] = dst
        return img
