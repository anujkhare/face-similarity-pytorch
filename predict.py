import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from src.models import segnet
from src import preprocess, dataset


def get_image_tensor(image_path: str, device) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Missing or corrupt image: {}'.format(image_path))
    
    image, _ = preprocess.resize_and_pad_image(image, 256)
    image = image.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
    return torch.from_numpy(image).to(device)


def predict(image_path1, image_path2, model, device):
    THRESH = 1.2
    images1 = get_image_tensor(image_path1, device)
    images2 = get_image_tensor(image_path2, device)

    pred = 0
    with torch.no_grad():
        feats1, feats2 = model(images1, images2)
        dist = torch.nn.functional.pairwise_distance(feats1, feats2)
        dist = dist.data.cpu().numpy()[0]
        if dist < THRESH:
            pred = 1
    return dist, pred


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: contains the named arguments parsed.
    """
    parser = argparse.ArgumentParser(description="Face Similarity PyTorch")
    parser.add_argument("-i1", "--image-path-1", help="Path to the first image", required=True, type=str,)
    parser.add_argument("-i2", "--image-path-2", help="Path to the second image", required=True, type=str,)
    parser.add_argument("-g", "--gpu", help="GPU ID to use. -1 for CPU.", required=False, type=int, default=-1)
    parser.add_argument("-w", "--weight-path", help="Path to the trained model weights.", required=False, type=str, default="face-siamese.pt")
    args = parser.parse_args()
    return args


def get_model(device, weight_path):
    model = segnet.SiameseNetworkLarge(256).to(device)
    model.train(False)
    model.eval()

    if not os.path.exists(weight_path):
        raise FileNotFoundError('The weights must be present at: {}'.format(weight_path))
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    return model


def main():
    args = parse_args()
    device = 'cpu'
    if args.gpu >= 0:
        device='cuda:{}'.format(args.gpu)

    model = get_model(device, args.weight_path)
    prob, pred = predict(args.image_path_1, args.image_path_2, model, device)
    print('Dis-similarity score: {:.2f}'.format(prob))
    if pred == 1:
        print('The two images are of the same person!')
    else:
        print('The two images are of different people!')

        
if __name__ == '__main__':
    main()