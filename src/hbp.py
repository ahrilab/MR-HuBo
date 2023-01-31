from pathlib import Path
from typing import List, Dict
from typing import Union
from scipy.spatial.transform import Rotation as R

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from colour import Color
from typing import Union
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import create_list_chunks
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file

from src.misc import reachy_to_smpl_idx

class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 num_betas: int = 16,
                 joint_idx: list = [i for i in range(21)]
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, num_betas=num_betas, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # self.bm.f
        self.n_joints = len(joint_idx)
        self.joint_idx = joint_idx
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(self.n_joints)])
            
    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts': new_body.Jtr[:, self.joint_idx], 'body': new_body}



def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray, rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)

    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        print('betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }




def run_ik_engine(res_path, motion, batch_size, smpl_path, vposer_dir, num_betas, device, verbosity):
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    optimizer_args = {'type': 'LBFGS', 'max_iter': 500, 'lr': 1, 'tolerance_change': 1e-4}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_dir,
                            verbosity=verbosity,
                            display_rc=(2, 2),
                            data_loss=data_loss,
                            num_betas=num_betas,
                            stepwise_weights=stepwise_weights,
                            optimizer_args=optimizer_args).to(device)
                            
    all_results = {}
    batched_frames = create_list_chunks(np.arange(len(motion)), batch_size, overlap_size=0, cut_smaller_batches=False)
    batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')

    for cur_frame_ids in batched_frames:
        target_pts = torch.from_numpy(motion[cur_frame_ids]).to(device).float()

        source_pts = SourceKeyPoints(bm=smpl_path, joint_idx=reachy_to_smpl_idx).to(device)

        ik_res = ik_engine(source_pts, target_pts, {})
        ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
        nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
        for k, v in ik_res_detached.items():
            if k not in all_results: all_results[k] = []
            all_results[k].append(v)

    d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
    d['betas'] = np.median(d['betas'], axis=0)

    transformed_d = transform_smpl_coordinate(bm_fname=smpl_path, 
                                            trans=d['trans'], 
                                            root_orient=d['root_orient'],
                                            betas=d['betas'], 
                                            rotxyz=[90, 0, 0])
    d.update(transformed_d)     
    d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])], axis=1)
    d['surface_model_type'] = 'smplx'
    d['gender'] = 'neutral'
    d['mocap_frame_rate'] = 30
    d['num_betas'] = num_betas
    np.savez(res_path, **d)

    return d



def make_vids(vid_path, smpl_data, len_motion, smpl_path, num_betas, fps):
    bm = BodyModel(bm_fname=smpl_path, num_betas=num_betas)
    smpl_dict = np.load(smpl_path)
    mean_pose_hand = np.repeat(np.concatenate([smpl_dict['hands_meanl'], smpl_dict['hands_meanr']])[None], axis=0, repeats=len_motion)

    body_parms = {**smpl_data, 'betas': np.repeat(smpl_data['betas'][None], axis=0, repeats=len_motion), 'pose_hand':mean_pose_hand}
    body_parms = {k:torch.from_numpy(v) for k,v in body_parms.items() if k in ['root_orient', 'trans', 'pose_body', 'pose_hand']}

    img_array = render_smpl_params(bm, body_parms, [-90, 0, 0])[None, None]
    imagearray2file(img_array, outpath=vid_path, fps=fps)
