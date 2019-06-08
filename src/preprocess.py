import cv2
import numpy as np
from typing import Tuple


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