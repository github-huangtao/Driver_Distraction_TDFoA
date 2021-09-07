#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.1
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: 基于DFoA和TDFoA生成评价指标值，生成csv文件
@author: Huang Tao
"""
import numpy as np
import cv2
import os
import scipy.io
import scipy.ndimage
from scipy.ndimage import filters
import pandas as pd
from evaluation_metrics import kld_numeric,cc_numeric,sim
import glob


predict_TDFoA_path='C:/Users/lenovo/Desktop/222/Generate_TDFoA/'
save_path='C:/Users/lenovo/Desktop/222/Generate_TDFoA/Compare_T_P.csv'
predict_TDFoA = glob.glob(predict_TDFoA_path+'/*')
TDFoA = [xx.replace('Generate_TDFoA', 'maps') for xx in predict_TDFoA]
metric_value=[]
for i in range(len(predict_TDFoA)):
    predict_TDFoA_map=cv2.imread(predict_TDFoA[i])
    TDFoA_map=cv2.imread(TDFoA[i])
    sim_value = round(sim(TDFoA_map, predict_TDFoA_map), 3)
    cc_value = round(cc_numeric(TDFoA_map, predict_TDFoA_map), 3)
    kld_value = round(kld_numeric(TDFoA_map, predict_TDFoA_map), 3)
    data=[sim_value,cc_value,kld_value]
    metric_value.append(data)
metric_value=np.array(metric_value)
metric_value = pd.DataFrame(metric_value)
metric_value.to_csv(save_path,header=None,index=None)
