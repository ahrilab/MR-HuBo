import torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def rotmat2rep(rotmat):
    res = matrix_to_rotation_6d(torch.from_numpy(rotmat)).numpy()
    return res


def rep2rotmat(rep):
    res = rotation_6d_to_matrix(torch.from_numpy(rep)).numpy()
    return res


def quat2rep(quat):
    res = quaternion_to_matrix(torch.from_numpy(quat))
    res = matrix_to_rotation_6d(res).numpy()
    return res
