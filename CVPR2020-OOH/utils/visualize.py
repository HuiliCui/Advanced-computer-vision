import cv2
import os
from datautils import load_json
from smpl_torch_batch import SMPLModel
import numpy as np
import torch
from datautils import draw_keypoints, draw_mask, draw_bbox
import random
import argparse

def main(args):

    base_dir = args.base_dir
    dataset = args.dataset
    dataset_dir = os.path.join(base_dir, dataset)
    annot = load_json(os.path.join(dataset_dir, 'annots.json'))
    model = SMPLModel(model_path='./data/model.pkl')

    samples = random.sample(range(1, len(annot)), 100)

    for item in samples:
        item = '%05d' %item
        param = annot[item]

        # load SMPL parameters
        pose = np.array(param['pose'])
        shape = np.array(param['betas'])
        scale = np.array(param['scale'])
        trans = np.array(param['trans'])

        # load camera parameters
        intri = np.array(param['intri'])
        extri = np.array(param['extri'])

        # load image
        img_path = os.path.join(dataset_dir, param['img_path'])
        img = cv2.imread(img_path)

        shape = torch.from_numpy(shape).type(torch.FloatTensor).resize(1, 10)
        pose = torch.from_numpy(pose).type(torch.FloatTensor).resize(1, 72)
        trans = torch.from_numpy(trans).type(torch.FloatTensor).resize(1, 3)
        scale = torch.from_numpy(scale).type(torch.FloatTensor)
        mesh, joints = model(shape, pose, trans, scale)

        mesh = mesh.numpy().reshape(6890, 3)
        joints = joints.numpy().reshape(24, 3)
        # We also provide 2D keypoints in the format used by LSP.
        # kp_2d = np.array(param['lsp_joints_2d'])

        # draw mask
        if dataset == 'trainset':
            img = draw_mask(dataset_dir, param, img)
        # draw keypoints
        im = draw_keypoints(joints, extri, intri, img)
        # draw bbox
        im = draw_bbox(param, im)

        # visualize
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("sample",0)
        cv2.resizeWindow("sample",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("sample",0,0)
        cv2.imshow('sample',im/255.)
        print(img_path)
        cv2.waitKey()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visulization of 3DOH50K")
    parser.add_argument('--base_dir', type=str, default='./', help='the path of 3DOH50K dataset')
    parser.add_argument('--dataset', type=str, default='trainset')
    args = parser.parse_args()
    main(args)
        
