#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 05.15
Implemented using TensorFlow 1.15 and Keras 2.2.4
function: Ideal Driver attention Prediction
@author: Huang Tao

数据集的制作（train,val,test）
根据json_files将数据划分成train,val,test
"""


import os
import json
from shutil import copy

DADA_path = 'K:/driver_attention/DADA-2000/DADA_dataset/'  # change to your own path

link_paths = ['I:/DADA2020/train/',
              'I:/DADA2020/test/',
              'I:/DADA2020/val/',
              ]  # change to your own path, and in the same order as the json file below.

json_files = ["K:/driver_attention/DADA-2000/train_file.json",
              "K:/driver_attention/DADA-2000/test_file.json",
              "K:/driver_attention/DADA-2000/val_file.json",
              ]  # change to your own path

if __name__ == '__main__':
    for json_file, link_path in zip(json_files, link_paths):
        with open(json_file, 'r') as f:
            train_f = json.load(f)
            for file in train_f:
                print(file)
                save_p = os.path.join(link_path, file[1])
                src = os.path.join(DADA_path, file[0][0],file[0][1])
                src=os.path.join(src,'fixation/maps')
                save_p=os.path.join(save_p,'fixation/maps')
                isExists = os.path.exists(save_p)
                if not isExists:
                    os.mkdir(save_p)
                filelist = os.listdir(src)
                for file in filelist:
                    src_file=os.path.join(src, file)
                    save_p_file=os.path.join(save_p,file)
                    copy(src_file,save_p_file)
