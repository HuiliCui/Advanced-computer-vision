import numpy as np
import pickle
import os
import random
from time import time
from numpy.linalg import solve
import copy
import cv2

class UV_Map_Generator():
    def __init__(self, UV_height, UV_width=-1, 
        UV_pickle='param.pickle'):
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        
        ### Load UV texcoords and face mapping info
        with open(UV_pickle, 'rb') as f:
            tmp = pickle.load(f)
        for k in tmp.keys():
            setattr(self, k, tmp[k])

    def UV_interp(self, rgbs):
        face_num = self.vt_faces.shape[0]
        vt_num = self.texcoords.shape[0]
        assert(vt_num == rgbs.shape[0])
       # uvs = self.vts #self.texcoords * np.array([[self.h - 1, self.w - 1]])
        triangle_rgbs = rgbs[self.vt_faces][self.face_id]
        bw = self.bary_weights[:,:,np.newaxis,:]
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        return im

    def get_UV_map(self, verts):
        xmin = verts[:, 0].min()
        xmax = verts[:, 0].max()
        ymin = verts[:, 1].min()
        ymax = verts[:, 1].max()
        zmin = verts[:, 2].min()
        zmax = verts[:, 2].max()
        vmin = np.array([xmin, ymin, zmin])
        vmax = np.array([xmax, ymax, zmax])
        box = (vmax-vmin).max() #2019.11.9 vmax.max()
        verts = (verts - vmin) / box - 0.5
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = verts[vt_to_v_index]
        uv = self.UV_interp(rgbs)
        return uv, vmin, vmax, box

    def get_ocuv(self, uv, verts, proj_verts, mask, vmin, vmax):
        box = (vmax-vmin).max()
        for v in range(len(verts)):
            if int(proj_verts[v][0]) > 255 or int(proj_verts[v][1]) > 255 or int(proj_verts[v][0]) < 0 or int(proj_verts[v][1]) < 0:
                continue
            if mask[int(proj_verts[v][1])][int(proj_verts[v][0])].all() < 1e-2:
                verts[v] = vmin
        verts = (verts - vmin) / box - 0.5
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = verts[vt_to_v_index]
        oc_uv = self.UV_interp(rgbs)
        temp_ = uv - oc_uv
        oc_uv[np.where(temp_>0)] = -0.5
        return oc_uv

    
