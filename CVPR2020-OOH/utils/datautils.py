# -*- coding: utf-8 -*-
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import os

lsp_skeleton = [[0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
            [2, 12],
            [12, 8],
            [8, 7],
            [7, 6],
            [3, 12],
            [12, 9],
            [9, 10],
            [10, 11],
            [12, 13]]

smpl_skeleton = [[0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [4, 7],
            [2, 5],
            [5, 8],
            [3, 6],
            [6, 9],
            [9, 10],
            [9, 11],
            [9, 12],
            [10, 13],
            [12, 15],
            [11, 14],
            [14, 16],
            [16, 18],
            [15, 17],
            [17, 19],
            [10, 13]]

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(smpl_skeleton) + 2)]
colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

def load_json(path):
    with open(path) as f:
        param = json.load(f)
    return param

def draw_keypoints(joint, extrinsic, intrinsic, im):

    intri = np.insert(intrinsic,3,values=0.,axis=1)
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extrinsic, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)[[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21]]

    for i in range(len(smpl_skeleton)):
        line = smpl_skeleton[i]
        color = colors[i]
        im = cv2.line(im, tuple(out_point[line[0]]), tuple(out_point[line[1]]), color, 5)

    for i in range(len(out_point)):
        im = cv2.circle(im, tuple(out_point[i]), 10, (0,0,255),-1)

    return im

def draw_mask(dataset_dir, param, img):
    mask_path = os.path.join(dataset_dir, param['mask_path'])
    mask = cv2.imread(mask_path, 0)
    color_mask = np.random.randint(0, 256, (1,3), dtype=np.uint8)
    img[np.where(mask>0)] = img[np.where(mask>0)] * 0.5 + color_mask * 0.5

    return img

def draw_bbox(param, im):
    bbox = np.array(param['bbox'])
    im = cv2.rectangle(im,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[1][0]),int(bbox[1][1])),(0,0,255),5)

    return im





