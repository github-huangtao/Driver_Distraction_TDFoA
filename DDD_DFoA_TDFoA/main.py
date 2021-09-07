#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.01
Implemented using TensorFlow 1.15 and Keras 2.2.4
function:基于DFoA与TDFoA的驾驶员分心检测模型
@author: Huang Tao
"""


import pandas as pd
import numpy as  np
import keras
from keras.models import Model
from keras.layers import Dense,  BatchNormalization, Activation,Dropout,Input
from keras import backend as K
from keras.optimizers import Adam,Adadelta
from keras.regularizers import l2
from keras.callbacks import TensorBoard,  ReduceLROnPlateau
K.set_learning_phase(1)

#生成寻训练数据的形式,分心类别为0，不分心类别为1
def generate_data(distraction_path,non_distraction_path):
    data_x = []
    data_y = []
    distraction_data = pd.read_csv(distraction_path, header=None)
    non_distraction_data = pd.read_csv(non_distraction_path, header=None)
    for i in range(len(distraction_data)):
        data_x.append(distraction_data.iloc[i, :])
        data_y.append(0)
    for i in range(len(non_distraction_data)):
        data_x.append(non_distraction_data.iloc[i, :])
        data_y.append(1)
    data_x=np.array(data_x)
    # data_y=np.array(data_y)
    data_y = keras.utils.to_categorical(data_y, 2)

    return data_x,data_y
def DDD_model():
    input_shape = (3,)
    inputs = Input(shape=input_shape)
    outputs = Dense(2, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    train_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/train/distraction.csv'
    train_non_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/train/non-distraction.csv'
    valid_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/valid/distraction.csv'
    valid_non_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/valid/non-distraction.csv'
    train_data_x, train_data_y = generate_data(train_distraction_path, train_non_distraction_path)
    valid_data_x, valid_data_y = generate_data(valid_distraction_path, valid_non_distraction_path)

    model = DDD_model()
    log_dir = 'logs/'
    logging = TensorBoard(log_dir=log_dir)  # 记录训练的损失与准确率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    model.fit(train_data_x, train_data_y, batch_size=128, epochs=100, verbose=1, validation_data=(valid_data_x, valid_data_y),callbacks=[logging, reduce_lr])

    model.save("models/our_DDD.h5")



if __name__ == '__main__':
    main()
