#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 06.15
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: 生成模型训练所需要的数据，该数据类型是一个生成器
@author: Huang Tao

"""
import numpy as np
import os
from configs import *
import glob
import random
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps


def process_test_data(test_pth):
    Xims = np.zeros((1, len(test_pth), input_shape[0], input_shape[1], 3))
    X = preprocess_images(test_pth, input_shape[0], input_shape[1])
    Xims[0, 0:len(test_pth), :] = np.copy(X)
    return Xims  #

def generator_data(video_b_s, phase_gen='train'):
    num_frames = input_t
    if phase_gen == 'train':
        train_pth = os.path.join(train_data_pth, '*', 'images')
        images_seq = glob.glob(train_pth)
        datas = []
        for image_pth in images_seq:
            images = sorted(glob.glob(image_pth + '/*'))
            maps = [xx.replace('images', 'maps') for xx in images]
            fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    [maps[jj + input_t], ],  #注意力选择序列的最后一个
                    [fixs[jj + input_t], ]   #选择序列的最后一个
                ))
        counts = 0
        print('len -> train data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas)-video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas)-video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][1], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(datas[counts][2], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Ymaps[i, :] = np.copy(Y[0])
                Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1
            yield [Xims], [Ymaps, Ymaps, Yfixs]  #

    else:
        val_pth = os.path.join(val_data_pth, '*', 'images')
        images_seq = glob.glob(val_pth)

        datas = []
        for image_pth in images_seq:
            images = sorted(glob.glob(image_pth + '/*'))
            maps = [xx.replace('images', 'maps') for xx in images]
            fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> val data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas) - video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas) - video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][1], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(datas[counts][2], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Ymaps[i, :] = np.copy(Y[0])
                Yfixs[i, :] = np.copy(Y_fix[0])
                counts += 1

            yield [Xims], [Ymaps, Ymaps, Yfixs]  #
if __name__ == '__main__':
    a = generator_data(5, phase_gen='train')
    for i in a:
        a = np.array(i[0])
        print(a.shape)
