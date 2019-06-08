import dlib
import numpy as np
import torchvision
from PIL import Image


class FaceCropTransform:
    def __init__(self, margin: float = 0.2) -> None:
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

        return image[top: bottom + 1, left: right + 1]


def get_transforms_inference():
    return torchvision.transforms.Compose([
        FaceCropTransform(margin=0.2),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((160, 160)),
    ])


def get_transforms_train(image_side):
    return torchvision.transforms.Compose([
        FaceCropTransform(margin=0.2),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=(-10, 10), resample=Image.BILINEAR),
        torchvision.transforms.Resize((image_side, image_side)),
    ])
