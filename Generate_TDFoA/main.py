#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 08.20
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: Ideal Driver attention Prediction
@author: Huang Tao

"""



from __future__ import division

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets import my_net
from configs import *
from data_processing import process_test_data, generator_data
from loss_function import kl_loss, cc_loss, nss_loss
import tqdm
from utilities import postprocess_predictions
from scipy.misc import imread, imsave
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import json
import os
import scipy.io as sio
import numpy as np
import random
from scipy.misc import imread, imresize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


if __name__ == '__main__':
    if mode == 'train':
        stateful = False
        x = Input(batch_shape=(batch_size, input_t, input_shape[0], input_shape[1], 3))
        m = Model(inputs=x, outputs=my_net(x, stateful))
        print("Compiling My_Net")
        m.compile(Adam(lr=1e-4), loss=[kl_loss, cc_loss, nss_loss], loss_weights=[1, 0.1, 0.1])  #
        print("Training My_Net")

        if pre_train:
            m.load_weights(pre_train_path)
        #DADA数据集
        train_generator=generator_data(video_b_s=batch_size, phase_gen='train')
        valid_generator=generator_data(video_b_s=batch_size, phase_gen='val')
        logging = TensorBoard(log_dir=log_dir)  # 记录训练的损失与准确率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
        m.fit_generator(train_generator, nb_train, epochs=nb_epoch,validation_data=valid_generator,
                        validation_steps=nb_videos_val, callbacks=[logging, reduce_lr])
        m.save("models/our.h5")
