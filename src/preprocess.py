import cv2
import numpy as np
from typing import Tuple
from PIL import Image
import dlib
import numpy as np


class FaceCropTransform:
    def __init__(self, margin=0.2):
        self.detector = dlib.get_frontal_face_detector()
        self.margin = margin
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        margin = self.margin
        rects = self.detector(image)

        # If no face detected, just use the original image
        if len(rects) == 0:
            return image

        if len(rects) > 1:
#             return image
            # Take the max area rect
            rects = sorted(rects, reverse=True, key=lambda x: x.area())

        rect = rects[0]
        top, left, right, bottom = rect.top(), rect.left(), rect.right(), rect.bottom()
        h, w = bottom - top + 1, right - left + 1

        # Add a percentage margin
        top = max(0, top - int(h * margin))
        bottom = min(H, bottom + int(h * margin))
        left = max(0, left - int(w * margin))
        right = min(W, right + int(w * margin))

        return image[top: bottom+1, left: right+1]


def resize_and_pad_image(image: np.ndarray, max_h) -> Tuple[np.ndarray, float]:
    # resize the image
    scale = max_h / image.shape[0]
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)

    # pad to 32
    h, w = image.shape[:2]
    dh = 32 - h%32
    dw = 32 - w%32
    dh, dw = list(map(lambda x: x if x<32 else 0,[dh, dw]))
    image = np.pad(image, ((0,dh),(0,dw),(0,0)),mode="constant",constant_values=255)
    return image, scale