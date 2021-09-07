#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 05.20
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: TDFoA模型
@author: Huang Tao

"""
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from keras.models import Model
from keras.layers import MaxPooling3D, Conv2D, GlobalAveragePooling2D, Concatenate, Lambda, ConvLSTM2D, Conv3D
from keras.layers import TimeDistributed, Multiply, Add, UpSampling2D, BatchNormalization, ReLU, Dropout
from configs import *
from keras.layers import Input, Layer, Dense,Lambda
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from layers import residual_attention_U2Net,residual_attention_U2Net_2D
#定义Mish激活函数
class Mish(Layer):
    def __init__(self,**kwargs  ):
        super().__init__(**kwargs)

    def forward(self, x):
        return x * (K.tanh(K.softplus(x)))
#特征提取结构
def feature_extractor(shapes=(batch_size, input_t, input_shape[0], input_shape[1], 3)):
    inputs = Input(batch_shape=shapes)
    x = Conv3D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,32,downsample=False)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,64,downsample=True)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,64,downsample=False)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,128,downsample=True)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,128,downsample=False)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,256,downsample=True)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x=residual_attention_U2Net(x,1,256,downsample=False)
    x = BatchNormalization()(x)
    x = Mish()(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def my_net(x, stateful=False):
    encoder = feature_extractor()
    x = encoder(x)
    outs = ConvLSTM2D(filters=256, kernel_size=3, padding='same', stateful=stateful)(x)
    outs=residual_attention_U2Net_2D(outs,1,128,downsample=False)
    outs = BatchNormalization()(outs)
    outs = Mish()(outs)
    outs = UpSampling2D(2, interpolation='bilinear')(outs)
    outs=residual_attention_U2Net_2D(outs,1,64,downsample=False)
    outs = UpSampling2D(2, interpolation='bilinear')(outs)
    outs=residual_attention_U2Net_2D(outs,1,32,downsample=False)
    outs = UpSampling2D(2, interpolation='bilinear')(outs)
    outs=residual_attention_U2Net_2D(outs,1,1,downsample=False)
    outs=Conv2D(filters=1, kernel_size=3, strides=1, padding='same',activation='sigmoid')(outs)

    return outs, outs, outs


if __name__ == '__main__':
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    session = tf.Session(config=config)  # 设置session KTF.set_session(sess)

    x = Input(batch_shape=(32, 5, 256, 192, 3))
    m = Model(inputs=x, outputs=my_net(x))
    print("Compiling MyNet")
    m.summary()

