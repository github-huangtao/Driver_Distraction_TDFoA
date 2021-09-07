#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.1
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: 损失函数的定义
@author: Huang Tao
"""


import numpy as np
import keras.backend as K

def kl_loss(y_true, y_pred, eps=K.epsilon()):
    """
    Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

    kld = K.sum(Q * K.log(eps + Q / (eps + P)), axis=[1, 2, 3])

    return kld


def information_gain(y_true, y_pred, y_base, eps=K.epsilon()):
    """
    Information gain (sec 4.1.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param y_base: baseline.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    B = y_base

    Qb = K.round(Q)  # discretize at 0.5
    N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

    ig = K.sum(Qb * (K.log(eps + P) / K.log(2) - K.log(eps + B) / K.log(2)), axis=[1, 2, 3]) / (K.epsilon() + N)

    return ig


def nss_loss(y_true, y_pred):
    """
    Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true

    Qb = K.round(Q)  # discretize at 0.5
    N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

    mu_P = K.mean(P, axis=[1, 2, 3], keepdims=True)
    std_P = K.std(P, axis=[1, 2, 3], keepdims=True)
    P_sign = (P - mu_P) / (K.epsilon() + std_P)

    nss = (P_sign * Qb) / (K.epsilon() + N)
    nss = K.sum(nss, axis=[1, 2, 3])

    return -nss  # maximize nss


def cc_loss(y_true, y_pred):
    eps = K.epsilon()
    P = y_pred
    P = P / (eps + K.sum(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    Q = Q / (eps + K.sum(Q, axis=[1, 2, 3], keepdims=True))

    N = y_pred._shape_as_list()[1] * y_pred._shape_as_list()[2]

    E_pq = K.sum(Q * P, axis=[1, 2, 3], keepdims=True)
    E_q = K.sum(Q, axis=[1, 2, 3], keepdims=True)
    E_p = K.sum(P, axis=[1, 2, 3], keepdims=True)
    E_q2 = K.sum(Q ** 2, axis=[1, 2, 3], keepdims=True) + eps
    E_p2 = K.sum(P ** 2, axis=[1, 2, 3], keepdims=True) + eps

    num = E_pq - ((E_p * E_q) / N)
    den = K.sqrt((E_q2 - E_q ** 2 / N) * (E_p2 - E_p ** 2 / N))

    return K.sum(- (num + eps) / (den + eps), axis=[1, 2, 3])  # 相关系数。|cc|<=1, =0 则不相关 1 则正相关， -1 则表示负相关
