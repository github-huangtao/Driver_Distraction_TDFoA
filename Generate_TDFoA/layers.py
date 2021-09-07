#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 05.19
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: 基于软阈值的的注意力、encoder-decoder的残差结构、Mish激活函数
@author: Huang Tao

"""

from keras.layers import MaxPooling3D, GlobalAveragePooling3D, Concatenate, Lambda, Conv3D,AveragePooling3D,Dense,Conv2D,MaxPooling2D
from keras.layers import Multiply, Add, UpSampling3D, BatchNormalization,UpSampling2D,GlobalAveragePooling2D
import keras.backend as K
import tensorflow as tf
from keras.layers import Input
from keras.regularizers import l2
from keras.engine import Layer
import keras
from keras.models import Model
from configs import *


#定义Mish激活函数
class Mish(Layer):
    def __init__(self,**kwargs  ):
        super().__init__(**kwargs)

    def forward(self, x):
        return x * (K.tanh(K.softplus(x)))

def abs_backend(inputs):
    return K.abs(inputs)
def expand_dim_backend(inputs):
    x = K.expand_dims(inputs, 1)
    x = K.expand_dims(x, 1)
    x = K.expand_dims(x, 1)
    return x
def expand_dim_backend_2D(inputs):
    x = K.expand_dims(inputs, 1)
    x = K.expand_dims(x, 1)
    return x
def sign_backend(inputs):
    return K.sign(inputs)
def new_Conv3D(residual,out_channels,k,s,d):
    residual = Conv3D(filters=out_channels, kernel_size=k, strides=s,dilation_rate=d, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(residual)
    residual = BatchNormalization()(residual)
    residual = Mish()(residual)
    return residual
def new_Conv2D(residual,out_channels,k,s,d):
    residual = Conv2D(filters=out_channels, kernel_size=k, strides=s,dilation_rate=d, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(residual)
    residual = BatchNormalization()(residual)
    residual = Mish()(residual)
    return residual
def upsample_like(src, tar):
    h = int(tar.shape[2]//src.shape[2])
    w = int(tar.shape[3]//src.shape[3])
    src = UpSampling3D((1,h,w))(src)
    return src

def upsample_like2d(src, tar):
    h = int(tar.shape[1]//src.shape[1])
    w = int(tar.shape[2]//src.shape[2])
    src = UpSampling2D((h,w))(src)
    return src
#基于深度可分离卷积的线性瓶颈深度残差收缩结果
def residual_attention_U2Net(incoming, nb_blocks,out_channels,downsample=False):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    for i in range(nb_blocks):
        identity = residual
        if downsample:
            Encond0 = new_Conv3D(residual, out_channels, 3, (1,2,2), 1)
        else:
            Encond0 = new_Conv3D(residual,out_channels,3,1,1)
        Encond1 = new_Conv3D(Encond0,12,3,1,1)
        Encond1_M = MaxPooling3D((1, 2, 2))(Encond1)
        Encond2 = new_Conv3D(Encond1_M,12,3,1,1)
        Encond2_M = MaxPooling3D((1, 2, 2))(Encond2)
        Encond3 = new_Conv3D(Encond2_M, 12, 3, 1,1)

        x = new_Conv3D(Encond3, 12, 3, 1,2)

        concat_x_E3 = Concatenate(axis=-1)([x,Encond3])
        Decond3=new_Conv3D(concat_x_E3,12,3,1,1)
        Decond3=upsample_like(Decond3,Encond2)
        concat_D3_E2=Concatenate(axis=-1)([Decond3, Encond2])
        Decond2 = new_Conv3D(concat_D3_E2, 12, 3, 1, 1)
        Decond2 = upsample_like(Decond2, Encond1)

        concat_D2_E1 = Concatenate(axis=-1)([Decond2, Encond1])
        Decond1 = new_Conv3D(concat_D2_E1, out_channels, 3, 1, 1)


        # Calculate global means
        Decond1_abs = Lambda(abs_backend)(Decond1)
        abs_mean = GlobalAveragePooling3D()(Decond1_abs)
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Mish()(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = keras.layers.subtract([Decond1_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(Decond1), n_sub])
        #下采样
        if downsample:
            identity = AveragePooling3D(pool_size=(1, 1, 1), strides=(1, 2,2))(identity)
        if in_channels != out_channels:
            identity = new_Conv3D(residual,out_channels,3,1,1)
        residual = keras.layers.add([residual, identity])
    return residual

def residual_attention_U2Net_2D(incoming, nb_blocks,out_channels,downsample=False):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    for i in range(nb_blocks):
        identity = residual
        if downsample:
            Encond0 = new_Conv2D(residual, out_channels, 3, (2,2), 1)
        else:
            Encond0 = new_Conv2D(residual,out_channels,3,1,1)
        Encond1 = new_Conv2D(Encond0,12,3,1,1)
        Encond1_M = MaxPooling2D(( 2, 2))(Encond1)
        Encond2 = new_Conv2D(Encond1_M,12,3,1,1)
        Encond2_M = MaxPooling2D(( 2, 2))(Encond2)
        Encond3 = new_Conv2D(Encond2_M, 12, 3, 1,1)

        x = new_Conv2D(Encond3, 12, 3, 1,2)

        concat_x_E3 = Concatenate(axis=-1)([x,Encond3])
        Decond3=new_Conv2D(concat_x_E3,12,3,1,1)
        Decond3=upsample_like2d(Decond3,Encond2)
        concat_D3_E2=Concatenate(axis=-1)([Decond3, Encond2])
        Decond2 = new_Conv2D(concat_D3_E2, 12, 3, 1, 1)
        Decond2 = upsample_like2d(Decond2, Encond1)

        concat_D2_E1 = Concatenate(axis=-1)([Decond2, Encond1])
        Decond1 = new_Conv2D(concat_D2_E1, out_channels, 3, 1, 1)


        # Calculate global means
        Decond1_abs = Lambda(abs_backend)(Decond1)
        abs_mean = GlobalAveragePooling2D()(Decond1_abs)
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Mish()(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend_2D)(scales)

        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = keras.layers.subtract([Decond1_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(Decond1), n_sub])
        #下采样
        if downsample:
            identity = AveragePooling2D(pool_size=( 1, 1), strides=( 2,2))(identity)
        if in_channels != out_channels:
            identity = new_Conv2D(residual,out_channels,3,1,1)
        residual = keras.layers.add([residual, identity])
    return residual

#
if __name__ == '__main__':
    inputs = Input(batch_shape=(32, 128, 96, 3))
    x=residual_attention_U2Net_2D(inputs, 1, 64, downsample=False)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
