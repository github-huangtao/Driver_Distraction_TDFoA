#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2021, 09.1
function: 提取视频中的图片（主要包括场景视频和其相对应的注意力图）
@author: Huang Tao
"""

import glob
import os
import tqdm
import cv2

#视频转图片
def video2image(video_pth, save_pth, mode='images'):
    vc = cv2.VideoCapture(video_pth)  #
    c = 0
    rval = vc.isOpened()
    while rval:  #
        c = c + 1
        rval, frame = vc.read()
        if rval:
            name = os.path.join(save_pth, str(c).zfill(4) + '.jpg')
            if mode == 'maps':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(name, frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()


if __name__ == '__main__':
    videos_pth = 'K:/driver_attention/DADA-2000/videos/*'  # Change to your own path
    maps_pth = 'K:/driver_attention/DADA-2000/maps/*'  # Change to your own path
    DADA_dataset_pth = 'I:/DADA/'  # Change to your own path of DADA_dataset
    all_videos = glob.glob(videos_pth)
    all_maps = glob.glob(maps_pth)

    for video in all_videos:
        video_name = os.path.basename(video)
        temp = video_name.split('_')
        cc, category, folder = temp[0], temp[1], temp[2].split('.')[0]
        save_images_pth = os.path.join(DADA_dataset_pth, category, folder, cc)
        if not os.path.exists(save_images_pth):
            os.makedirs(save_images_pth)
        video2image(video, save_images_pth)

    for maps in all_maps:  # If you  want to use maps after video conversion, Please cancel these comments..
        print(maps)
        maps_name = os.path.basename(maps)
        temp = maps_name.split('_')
        cc, category, folder = temp[0], temp[1], temp[2].split('.')[0]
        save_maps_pth = os.path.join(DADA_dataset_pth, category, folder, cc)
        if not os.path.exists(save_maps_pth):
            os.makedirs(save_maps_pth)
        video2image(maps, save_maps_pth, 'maps')
