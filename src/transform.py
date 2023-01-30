import numpy as np
import torch
from transforms3d.quaternions import quat2mat, mat2quat
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

def rotmat2rep(rotmat):
    # if len(rotmat.shape) != 2:
    #     raise ValueError('rotmat should be matrix')
    # if rotmat.shape[0] != 3 or rotmat.shape[1] != 3:
    #     raise ValueError('rotmat shape should be 3 x 3. current: ', rotmat.shape) 
    res = matrix_to_rotation_6d(torch.from_numpy(rotmat)).numpy()

    return res

def rep2rotmat(rep):
    # if len(rep.shape) != 2:
    #     raise ValueError('rep should be matrix')
    # if rep.shape[0] != 3 or rep.shape[1] != 2:
    #     raise ValueError('rep shape should be 3 x 2. current: ', rep.shape) 
    
    # rotmat = np.zeros((3, 3))
    # rotmat[:, 0] = rep[:, 0] / np.linalg.norm(rep[:, 0])
    # rotmat[:, 1] = rep[:, 1] - np.sum(rotmat[:, 0] * rep[:, 1])*rotmat[:, 0]
    # rotmat[:, 1] = rotmat[:, 1] / np.linalg.norm(rotmat[:, 1])
    # rotmat[:, 2] = np.cross(rotmat[:, 0], rotmat[:, 1]).T
    res = rotation_6d_to_matrix(torch.from_numpy(rep)).numpy()
    return res

def quat2rep(quat):
    res = quaternion_to_matrix(torch.from_numpy(quat))
    res = matrix_to_rotation_6d(res).numpy()
    return res

# def rep2quat(rep):
#     rotmat = rep2rotmat(rep)
#     quat = mat2quat(rotmat)
#     return quat