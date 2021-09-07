
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.1
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: TDFoA模型测试(图片)
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
t=5
pre_train_path="C:/Users/lenovo/Desktop/Ablation/trained_ablation_model/our/our.h5"
m=load_model(pre_train_path,custom_objects={'Mish':Mish},compile=False)
def process_test_data(test_pth):
    Xims = np.zeros((1, len(test_pth), 256, 192, 3))
    X = preprocess_images(test_pth, 256, 192)
    Xims[0, 0:len(test_pth), :] = np.copy(X)
    return Xims  #

test_path='C:/Users/lenovo/Desktop/222/image/'
save_paths='C:/Users/lenovo/Desktop/222/Generate_TDFoA/'

imgs = os.listdir(test_path)
imgs.sort()
imgs = [os.path.join(test_path, xx) for xx in imgs]
original_image = imread(imgs[0])
original_size = original_image.shape[1], original_image.shape[0]

for i in range(len(imgs)-t):
    x_in = process_test_data(imgs[i: i + t])
    pre_map = m.predict(x=x_in, batch_size=1)
    pre = pre_map[-1][0, :, :, 0]
    res = postprocess_predictions(pre, original_image.shape[0], original_image.shape[1])
    res = res.astype(int)
    save_name = os.path.basename(imgs[i + t]).split('.')[0]
    save_name1=save_name+'.jpg'
    imsave(os.path.join(save_paths, save_name1), res)
