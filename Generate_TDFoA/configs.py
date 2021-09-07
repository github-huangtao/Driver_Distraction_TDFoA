#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 05.15
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: 定义一些模型参数
@author: Huang Tao

"""
batch_size = 64
nb_train = 49088    #根据实际情况改变
nb_epoch = 1
nb_videos_val = 6658  #根据实际情况改变
log_dir = 'logs/'
# -----
smooth_weight = 0.5
input_t = 5  # 表示一次输入多少张图片，时间序列
input_shape = (256, 192)

shape_r, shape_c = input_shape[0], input_shape[1]
shape_r_out, shape_c_out = input_shape[0], input_shape[1]


mode = 'train'  # 'train' or 'test'
train_data_pth = 'I:/DADA2020/train'
val_data_pth = 'I:/DADA2020/val'

pre_train=False



