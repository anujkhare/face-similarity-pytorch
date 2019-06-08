import cv2
import itertools
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, Tuple

from src import preprocess


class PairDataset:
    """
    Data set that loads images pair-wise for training the Siamese network.

    Given two images, the label is defined as:
        - 0: if the images belong to different people
        - 1: if the images belong to the same person

    We'll pick up pairs of images from the given set using the following strategy:
        1. Positive pair: randomly pick any positive pair from all the available samples
        2. Negative pair: for one of the images picked for the positive pair, find a negative pair and add
    """

    @staticmethod
    def _read_image(p: str) -> np.ndarray:
        image = cv2.imread(str(p))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _image(self, ix: int) -> np.ndarray:
        image = self.images[ix]
        image = np.array(self.transforms(image))
        image = image.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
        return image

    def __init__(self, df: pd.DataFrame, image_side: int = 160) -> None:
        df.reset_index(inplace=True, drop=True)
        df.idx = df.index

        # Get all the positive pairs in the dataset
        idxs = df.groupby('label').idx.apply(list).values
        idx_by_person = list(filter(lambda x: len(x) > 1, idxs))

        pos_pairs = []
        for list_idx_of_person in idx_by_person:
            pairs = itertools.permutations(list_idx_of_person, 2)  # order matters here
            pos_pairs.extend(pairs)
        np.random.shuffle(pos_pairs)

        self.df = df
        self.images = [self._read_image(p) for p in df.path]
        self.labels = df.label.values
        self.pos_pairs = pos_pairs

        self.transforms = preprocess.get_transforms_train(image_side=image_side)

    def __len__(self) -> int:
        return len(self.pos_pairs)

    def _sample_neg_pair(self, idx: int) -> Tuple[int, int]:
        label = self.labels[idx]
        neg_idx = np.random.choice(self.df.loc[self.df.label != label].index)
        return idx, neg_idx

    def __getitem__(self, ix: int) -> Dict[str, Any]:
        pos_pair = self.pos_pairs[ix]
        neg_pair = self._sample_neg_pair(pos_pair[0])
        images1 = np.vstack([self._image(pos_pair[0]), self._image(neg_pair[0])])
        images2 = np.vstack([self._image(pos_pair[1]), self._image(neg_pair[1])])
        labels = np.array([1, 0]).astype(np.float32)

        return {
            'images1': images1,
            'images2': images2,
            'labels': labels,
            'pairs': np.array([pos_pair, neg_pair])
        }


def get_dataloader(df: pd.DataFrame, image_side: int, batch_size: int, num_workers: int = 4) -> Tuple:
    dataset = PairDataset(df.copy(), image_side=image_side)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Training: {:,} total positive pairs {:,} mini batches'.format(len(dataset), len(dataloader)))
    return dataset, dataloader


def flatten(batch):
    bsz, n, c, h, w = batch['images1'].shape
    images1, images2, labels = batch['images1'].view(-1, c, h, w), batch['images2'].view(-1, c, h, w), batch[
        'labels'].view(-1)
    return images1, images2, labels
