import torch
from copy import deepcopy
import numpy as np

def vts_position(vts, UV_map, index, change_v, new_v_to_vt):
    s = 0.
    f = vts.astype(np.int32)
    coords = [
        (f[0], f[1]),
        (f[0], f[1] + 1),
        (f[0] + 1, f[1]),
        (f[0] + 1, f[1] + 1),
    ]
    P = [None] * 4
    for i, coord in enumerate(coords):
        P[i] = UV_map[coord[0], coord[1]]
        if UV_map[coord[0], coord[1]].all() != 0.:
            s = s + 1.
    if s == 4.:
        u = vts[0] - coords[0][0]
        v = vts[1] - coords[0][1]
        face_point = (1 - u) * (1 - v) * P[0] + (1 - u) * v * P[1] + u * (1 - v) * P[2] + u * v * P[3]
        return face_point
    elif s!= 0.0:
        face_point = (P[0]+P[1]+P[2]+P[3]) / s
        return face_point
    else:
        new_v_to_vt[change_v].remove(index)
        return np.array([500., 500., 500.], dtype=np.float)

def resample_np(uv_generator, UV_map):
    # vts = uv_generator.vts
    new_vts = uv_generator.refine_vts
    vt_3d = [None] * new_vts.shape[0]
    resmaple_vvt = uv_generator.resample_v_to_vt
    vt_3d = UV_map[new_vts.T[0], new_vts.T[1]]
    vt_3d = np.stack(vt_3d)
    opt_v_3d = vt_3d[resmaple_vvt]
    return opt_v_3d

def resample_torch(uv_generator, UV_map, device):
    uv = UV_map.to(device)#torch.from_numpy(UV_map).to(device)
    new_vts = uv_generator.refine_vts
    resmaple_vvt = uv_generator.resample_v_to_vt
    vt_3d = UV_map[new_vts.T[0], new_vts.T[1]]
    opt_v_3d = vt_3d[resmaple_vvt].to(device) 
    return opt_v_3d

def vts_position_torch(vts, UV_map, index, change_v, new_v_to_vt):
    s = 0.
    f = vts.int()
    coords = [
        (f[0], f[1]),
        (f[0], f[1] + 1),
        (f[0] + 1, f[1]),
        (f[0] + 1, f[1] + 1),
    ]
    P = [None] * 4
    zero_tensor = torch.zeros(3)
    for i, coord in enumerate(coords):
        P[i] = UV_map[coord[0], coord[1]]
        if not torch.equal(UV_map[coord[0], coord[1]], zero_tensor):
            s = s + 1.
    if s == 4.:
        u = vts[0] - coords[0][0]
        v = vts[1] - coords[0][1]
        face_point = (1 - u) * (1 - v) * P[0] + (1 - u) * v * P[1] + u * (1 - v) * P[2] + u * v * P[3]
        return face_point
    elif s!= 0.0:
        face_point = (P[0]+P[1]+P[2]+P[3]) / s
        return face_point
    else:
        new_v_to_vt[change_v].remove(index)
        return torch.Tensor([500., 500., 500.])