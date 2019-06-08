from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from math import ceil, floor
from skimage.util import montage
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision.transforms import Compose
from typing import *
import copy 
import cv2
import glob
import inspect
import itertools
import json
import math
import matplotlib.pyplot as plt 
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import random 
import scipy
import sklearn
import socket
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from src import dataset, preprocess, visualize

def train_iter(batch, model, optimizer, loss_func, device) -> float:
    optimizer.zero_grad()

    images1, images2, labels = dataset.flatten(batch)
    probs = model(images1.cuda(device), images2.cuda(device))

    error = loss_func(probs, labels.cuda(device))
    error.backward()
    optimizer.step()

    return error.data.cpu().numpy()

def evaluate(dataloader_val, model, loss, device, n_iters=2):
    model.eval()

    error = 0
    cm_total = np.zeros((2, 2))

    with torch.no_grad():
        for ix, batch in enumerate(dataloader_val):
            if ix >= n_iters:
                break

            images1, images2, labels = dataset.flatten(batch)
            probs = model(images1.cuda(device), images2.cuda(device))
            error += loss(probs, labels.cuda(device)).data.cpu().numpy()
            
            _, labels_pred = torch.max(probs, dim=1)
            cm_total += sklearn.metrics.confusion_matrix(labels.data.cpu().numpy(), labels_pred.data.cpu().numpy())

            # Plot the image
            visualize.visualize(batch, maxn=10)
            print('Prob', np.exp(probs[:, 1].data.cpu().numpy()))
            print('Pred', labels_pred)

    model.train(True)
    return error, cm_total