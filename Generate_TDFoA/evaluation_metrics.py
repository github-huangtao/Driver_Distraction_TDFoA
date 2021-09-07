
import numpy as np
import cv2
from scipy.ndimage import filters

def kld_numeric(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    eps = np.finfo(np.float32).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob

    kld = np.sum(Q * np.log(eps + Q / (eps + P)))

    return kld

def cc_numeric(y_true, y_pred):
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    eps = np.finfo(np.float32).eps

    cv2.normalize(y_pred, dst=y_pred, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(y_true, dst=y_true, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    y_pred = (y_pred - np.mean(y_pred)) / (eps + np.std(y_pred))
    y_true = (y_true - np.mean(y_true)) / (eps + np.std(y_true))

    cc = np.corrcoef(y_pred, y_true)

    return cc[0][1]

def sim(y_true, y_pred):
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)
    y_pred = y_pred / 255
    y_pred = (y_pred - np.min(y_pred.ravel())) / (np.max(y_pred.ravel()) - np.min(y_pred.ravel()))
    img = y_pred / np.sum(y_pred.ravel())
    y_pred = np.array(y_pred)
    y_true = (y_true - np.min(y_true.ravel())) / (np.max(y_true.ravel()) - np.min(y_true.ravel()))
    img1 = y_true / np.sum(y_true.ravel())
    img1 = np.array(img1)

    diff = np.minimum(img, img1)
    score = np.sum(diff)
    return score

