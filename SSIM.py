from scipy import signal
from scipy.ndimage.filters import convolve
from config import cfg
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def get_SSIM(prediction, truth):
    b, s, c, h, w = prediction.shape
    ssim_arr = np.zeros((b, s, 3))
    for i in range(b):
        for day in range(s):
            for k in range(3):
                ssim_arr[i, day, k] = ssim(prediction[i, day, k], truth[i, day, k], data_range=1)
    return ssim_arr