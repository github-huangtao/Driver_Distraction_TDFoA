#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.1
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: TDFoA模型测试
@author: Huang Tao
验证模型的准确率
"""
from keras.layers import Input
from keras.models import Model,load_model
from nets import my_net
from utilities import postprocess_predictions
import os
from scipy.misc import imread, imsave
import numpy as np
from traffic_data import preprocess_images
from nets import Mish
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
stateful = True
pre_train_path="C:/Users/lenovo/Desktop/Ablation/trained_ablation_model/our/our.h5"
m=load_model(pre_train_path,custom_objects={'Mish':Mish},compile=False)
# m.summary()
def process_test_data(test_pth):
    Xims = np.zeros((1, len(test_pth), 256, 192, 3))
    X = preprocess_images(test_pth, 256, 192)
    Xims[0, 0:len(test_pth), :] = np.copy(X)
    return Xims  #

def padding(img, shape_r=320, shape_c=192, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = imresize(img, (shape_r, new_cols))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = imresize(img, (new_rows, shape_c))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

video_path='C:/Users/lenovo/Desktop/images_5_001.avi'
vc = cv2.VideoCapture(video_path)
c = 0
i=0
rval = vc.isOpened()
ims = np.zeros((4, 256, 192, 3),dtype=np.uint8)
ims_orogin = np.zeros((4, 660, 1584, 3),dtype=np.uint8)

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    original_image =frame
    original_size = original_image.shape[1], original_image.shape[0]
    padded_image = padding(frame, 256, 192, 3)
    # cv2.imshow('1',padded_image)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    if c<4:
        ims[c]=padded_image
        ims_orogin[c] = frame

    else:
        Xims = np.zeros((1, 3, 256, 192, 3))
        imgs = ims[0:3, :, :, :]
        imgs = imgs[:, :, :, ::-1]
        Xims[0, 0:3, :] = np.copy(imgs)
        pre_map = m.predict(x=Xims, batch_size=1)
        pre = pre_map[-1][0, :, :, 0]
        mask = postprocess_predictions(pre, original_image.shape[0], original_image.shape[1])
        mask = np.uint8(1 * mask)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        # # image = cv2.imread(ims[3])
        superimposed_img = cv2.addWeighted(ims_orogin[3], 0.6, mask, 0.4, 0)
        cv2.imshow("image", superimposed_img)
        if cv2.waitKey(1) == ord('q'):
            break
        b=ims[3]
        ims[3] = padded_image
        ims[0] = ims[1]
        ims[1] = ims[2]
        ims[2] = b
        d = ims_orogin[3]
        ims_orogin[3] = frame
        ims_orogin[0] = ims_orogin[1]
        ims_orogin[1] = ims_orogin[2]
        ims_orogin[2] = d
    c = c + 1
cap.release()
cv.destroyAllWindows()