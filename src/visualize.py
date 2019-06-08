import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage
from src.dataset import flatten


def visualize(batch, maxn=float('inf'), figsize=(50, 20)) -> None:
    images1, images2, labels = flatten(batch)

    images1 = images1.data.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    images2 = images2.data.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    maxn = min(maxn, images1.shape[0])

    images = []
    for ix in range(maxn):
        images.append(images1[ix])
        images.append(images2[ix])

    print(labels)
    plt.figure(figsize=figsize)
    plt.imshow(montage(images, grid_shape=(maxn, 2), multichannel=True))
    plt.show()
