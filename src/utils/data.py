from torch.utils.data import dataloader, Dataset
import numpy as np
import os.path as osp
import pickle
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import matrot2aa, aa2matrot
from human_body_prior.tools.model_loader import load_model
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from torch.distributions import multivariate_normal
from scipy.stats import normaltest
import random

device = 'cuda'
vposer_dir = './data/vposer_v2_05'
smpl_path = './data/bodymodel/smplx/neutral.npz'

def draw(probs):
    val = random.random()
    csum = np.cumsum(probs)
    i = sum(1 - np.array(csum>=val, dtype=int))
    if i == 2000:
        return -1
    else:
        return i

def split_train_test(reachy_dir, human_dir, num, split_ratio=10, sample_vposer=True):
    if sample_vposer:
        vp, ps = load_model(vposer_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        vp = vp.to(device)

    test_num = num // split_ratio

    all_reachy_xyzs = {'train':[], 'test':[]}
    all_reachy_reps = {'train':[], 'test':[]}
    all_reachy_angles = {'train':[], 'test':[]}
    all_smpl_reps = {'train':[], 'test':[]}
    all_smpl_rots = {'train':[], 'test':[]}

    for idx in range(num):
        print(idx, '/', num)
        smpl = np.load(osp.join(human_dir, 'params_{:03}.npz'.format(idx)))
        smpl_pose_body = smpl['pose_body']
        curr_num = len(smpl_pose_body)
        if sample_vposer:
            z = vp.encode(torch.from_numpy(smpl_pose_body).to(device))
            z_mean = z.mean.detach().cpu() # 2000 32
            dim_z = z_mean.shape[1]
            dist = multivariate_normal.MultivariateNormal(loc=torch.zeros(dim_z), 
                                                          covariance_matrix=torch.eye(dim_z))
            z_prob = torch.exp(dist.log_prob(z_mean))
            z_prob = z_prob/torch.sum(z_prob)

            n_sample = 500
            sample_idx = []
            while len(sample_idx) < n_sample:
                i = draw(z_prob)
                if i not in sample_idx and i > 0:
                    sample_idx.append(i)

        smpl_aa = smpl_pose_body.reshape(curr_num, -1, 3)
        num_smpl_joints = smpl_aa.shape[1]

        smpl_rot = aa2matrot(torch.from_numpy(smpl_aa.reshape(curr_num*num_smpl_joints, 3)))        
        smpl_rep = matrix_to_rotation_6d(smpl_rot)

        smpl_rot = smpl_rot.reshape(curr_num, num_smpl_joints, 3, 3)
        smpl_rep = smpl_rep.reshape(curr_num, num_smpl_joints, 6)

        smpl_rot = smpl_rot.numpy().reshape(curr_num, -1)
        smpl_rep = smpl_rep.numpy().reshape(curr_num, -1)


        reachy_angle = pickle.load(open(osp.join(reachy_dir, 'angles_{:03}.pkl'.format(idx)), 'rb'))
        angle_chunk = []
        for ra in reachy_angle:
            values = []
            for k in sorted(list(ra.keys())):
                values.append(ra[k])
            angle_chunk.append(np.array(values))
        angle_chunk = np.asarray(angle_chunk)

        reachy_xyzrep = np.load(osp.join(reachy_dir, 'xyzs+reps_{:03}.npz'.format(idx)))
        reachy_xyzs = reachy_xyzrep['xyzs']
        reachy_reps = reachy_xyzrep['reps'] 

        reachy_xyzs = reachy_xyzs.reshape(curr_num, -1)
        reachy_reps = reachy_reps.reshape(curr_num, -1)

        if idx < test_num:
            target = 'test'
        else:
            target = 'train'

        if sample_vposer:
            all_reachy_xyzs[target].append(reachy_xyzs[sample_idx])
            all_reachy_reps[target].append(reachy_reps[sample_idx]) 
            all_reachy_angles[target].append(angle_chunk[sample_idx])
            all_smpl_reps[target].append(smpl_rep[sample_idx])
            all_smpl_rots[target].append(smpl_rot[sample_idx])
        else:
            all_reachy_xyzs[target].append(reachy_xyzs)
            all_reachy_reps[target].append(reachy_reps) 
            all_reachy_angles[target].append(angle_chunk)
            all_smpl_reps[target].append(smpl_rep)
            all_smpl_rots[target].append(smpl_rot)
        
    for target in ['test', 'train']:
        all_reachy_xyzs[target] = np.concatenate(all_reachy_xyzs[target], axis=0)
        all_reachy_reps[target] = np.concatenate(all_reachy_reps[target], axis=0)
        all_reachy_angles[target] = np.concatenate(all_reachy_angles[target], axis=0)
        all_smpl_reps[target] = np.concatenate(all_smpl_reps[target], axis=0)
        all_smpl_rots[target] = np.concatenate(all_smpl_rots[target], axis=0)

    return all_reachy_xyzs, all_reachy_reps, all_reachy_angles, all_smpl_reps, all_smpl_rots        


class R4Rdata(Dataset):
    def __init__(self, reachy_xyz, reachy_rep, reachy_angle, smpl_rep, smpl_rot):
        self.reachy_xyz = reachy_xyz
        self.reachy_rep = reachy_rep
        self.reachy_angle = reachy_angle
        self.smpl_rep = smpl_rep
        self.smpl_rot = smpl_rot

    def __len__(self):
        return len(self.smpl_rep)

    def __getitem__(self, idx):
        sample = dict()

        sample['reachy_xyz'] = self.reachy_xyz[idx]
        sample['reachy_rep'] = self.reachy_rep[idx]
        sample['reachy_angle'] = self.reachy_angle[idx]
        sample['smpl_rep'] = self.smpl_rep[idx]
        sample['smpl_rot'] = self.smpl_rot[idx]

        return sample




