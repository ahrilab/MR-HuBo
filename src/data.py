from torch.utils.data import dataloader, Dataset
import numpy as np
import os.path as osp
import pickle
import torch
from pytorch3d.transforms import axis_angle_to_matrix

def split_train_test(reachy_dir, human_dir, num, split_ratio=10):
    test_num = num // split_ratio

    all_reachy_xyzs = {'train':[], 'test':[]}
    all_reachy_reps = {'train':[], 'test':[]}
    all_smpl_reps = {'train':[], 'test':[]}
    all_smpl_rots = {'train':[], 'test':[]}

    for idx in range(num):
        reachy = np.load(osp.join(reachy_dir, 'xyzs+rots_{:03}.npz'.format(idx)))
        reachy_xyzs = reachy['xyzs']
        reachy_rots = reachy['rots'] # rots should be changed to rep TODO

        curr_num = len(reachy_xyzs)
        reachy_xyzs = reachy_xyzs.reshape(curr_num, -1)
        reachy_rots = reachy_rots.reshape(curr_num, -1)

        smpl = np.load(osp.join(human_dir, 'params_{:03}.npz'.format(idx)))
        smpl_pose_body = smpl['pose_body']
        smpl_aa = smpl_pose_body.reshape(curr_num, -1, 3)

        smpl_rot = axis_angle_to_matrix(torch.from_numpy(smpl_aa)).reshape(curr_num, -1)
        smpl_rot = smpl_rot.numpy()
        
        smpl_rep = axis_angle_to_matrix(torch.from_numpy(smpl_aa))[:, :, :, :-1].reshape(curr_num, -1)
        smpl_rep = smpl_rep.numpy()

        if idx < test_num:
            target = 'test'
        else:
            target = 'train'

        all_reachy_xyzs[target].append(reachy_xyzs)
        all_reachy_reps[target].append(reachy_rots) # rots are actualy reps. should be corr.
        all_smpl_reps[target].append(smpl_rep)
        all_smpl_rots[target].append(smpl_rot)
        
    for target in ['test', 'train']:
        all_reachy_xyzs[target] = np.concatenate(all_reachy_xyzs[target], axis=0)
        all_reachy_reps[target] = np.concatenate(all_reachy_reps[target], axis=0)
        all_smpl_reps[target] = np.concatenate(all_smpl_reps[target], axis=0)
        all_smpl_rots[target] = np.concatenate(all_smpl_rots[target], axis=0)

    return all_reachy_xyzs, all_reachy_reps, all_smpl_reps, all_smpl_rots        


class R4Rdata(Dataset):
    def __init__(self, reachy_xyz, reachy_rep, smpl_rep, smpl_rot):
        self.reachy_xyz = reachy_xyz
        self.reachy_rep = reachy_rep
        self.smpl_rep = smpl_rep
        self.smpl_rot = smpl_rot

    def __len__(self):
        return len(self.smpl_rep)

    def __getitem__(self, idx):
        sample = dict()

        sample['reachy_xyz'] = self.reachy_xyz[idx]
        sample['reachy_rep'] = self.reachy_rep[idx]
        sample['smpl_rep'] = self.smpl_rep[idx]
        sample['smpl_rot'] = self.smpl_rot[idx]

        return sample




