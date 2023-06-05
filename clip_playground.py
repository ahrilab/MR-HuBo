import joblib
import numpy as np
import torch
import pickle
import copy
from body_visualizer.tools.vis_tools import render_smpl_params, imagearray2file, show_image
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import matplotlib.pyplot as plt
from src.misc import smplx_jname, joint_range, ret_keys
from src.net import MLP
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import matrot2aa, aa2matrot
import pytorch_kinematics as pk
# import kinpy as kp

class Angle(torch.nn.Module):
    def __init__(self, roa, chain):
        super().__init__()
        self.target = torch.nn.Parameter(torch.Tensor(roa), requires_grad=True)
        self.chain = chain
        self.th = {}

    def forward(self):        
        for ki, k in enumerate(sorted(joint_range.keys())):
            self.th[k] = self.target[ki]

        fk_res = self.chain.forward_kinematics(self.th)
        fk_xyz = []

        fk_jtr_name = ['torso', 'left_camera', 'right_camera', 'l_shoulder', 'l_forearm', 'l_wrist2hand', 'left_tip', 'r_shoulder', 'r_forearm', 'r_wrist2hand', 'right_tip']
        for k in fk_jtr_name:
            fk_xyz.append(fk_res[k].get_matrix()[:, :-1, -1])
        fk_xyz = torch.cat(fk_xyz)

        return fk_xyz

    def get_angle(self):
        new_th = {}
        for k, v in self.th.items():
            new_th[k] = v.detach().cpu().numpy()
        return new_th

##########################################
device = 'cuda'

vposer_dir = './data/vposer_v2_05'
smpl_path = './data/bodymodel/smplx/neutral.npz'
dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 512

###########################################
vp, ps = load_model(vposer_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
vp = vp.to(device)

bm = BodyModel(bm_fname=smpl_path).to(device)

model_pre = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_xyzs+dim_reachy_reps, dim_hidden=dim_hidden).to(device)
model_post = MLP(dim_input=dim_reachy_xyzs+dim_reachy_reps, dim_output=dim_reachy_angles, dim_hidden=dim_hidden).to(device)

model_pre.load_state_dict(torch.load('./models/human2reachy_best_pre_v2.pth'))
model_post.load_state_dict(torch.load('./models/human2reachy_best_post_v2.pth'))

model_pre.eval()
model_post.eval()


###########################################
num_betas = 16
num_joints =21
num_plots = 9

# load human smpl result from pymaf
pymaf = joblib.load(open('./pymaf.pkl', 'rb'))
refined_angles = []

_, ax = plt.subplots(1, 1)
plt.ion()
plt.show()

for idx in range(250, pymaf['pose'].shape[0]):
    # read angle axis. convert it to rot.
    smpl_aa = pymaf['pose'][:, 3:66][idx:idx+1]
    smpl_aa = torch.from_numpy(smpl_aa)
    smpl_rot = aa2matrot(smpl_aa.reshape(1*num_joints, -1))
    smpl_rep = matrix_to_rotation_6d(smpl_rot)
    smpl_rep = smpl_rep.reshape(1, num_joints, 6).reshape(1, -1)

    bm_res = bm(**{'pose_body':smpl_aa.reshape(1, -1).to(device)})
    smpl_xyz = bm_res.Jtr.detach().cpu()[0]
    smpl_jt_idx = [12, 23, 24, 16, 18, 20, 29, 17, 19, 21, 44]

    # get reachy info
    with torch.no_grad():
        reachy_xyzrep = model_pre(smpl_rep.to(device))
        reachy_xyz = reachy_xyzrep[:, :dim_reachy_xyzs]
        reachy_rep = reachy_xyzrep[:, dim_reachy_xyzs:]
        reachy_angle = model_post(reachy_xyzrep)[:, :dim_reachy_angles]

    # reachy_xyz = reachy_xyz.reshape(-1, 3).detach().cpu()

    # reachy angle: learnable parameter로 놓는다
    chain = pk.build_chain_from_urdf(open('./reachy.urdf').read()).to(device=device)
    angle = Angle(reachy_angle[0], chain)
    optimizer = torch.optim.AdamW(angle.parameters(), lr=0.1)


    for _ in range(100):
        fk_xyz = angle()
        fk_xyz[:, 1] = fk_xyz[:, 1] - fk_xyz[0, 1] + smpl_xyz[12, 0].to(device)
        fk_xyz[:, 2] = fk_xyz[:, 2] - fk_xyz[0, 2] + smpl_xyz[12, 1].to(device)
        fk_xyz[:, 0] = fk_xyz[:, 0] - fk_xyz[0, 0] + smpl_xyz[12, 2].to(device)

        loss = torch.sum((fk_xyz[:, 1] - smpl_xyz[smpl_jt_idx][:, 0].to(device))**2) + \
                torch.sum((fk_xyz[:, 2] - smpl_xyz[smpl_jt_idx][:, 1].to(device))**2) + \
                torch.sum((fk_xyz[:, 0] - smpl_xyz[smpl_jt_idx][:, 2].to(device))**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ax.clear()
        for ci, xyz in enumerate(smpl_xyz):
            if ci in smpl_jt_idx:
                ax.plot(xyz[0], xyz[1], 'ro')
                ax.text(xyz[0], xyz[1], str(ci))
        for ci, xyz in enumerate(fk_xyz):
            ax.plot(xyz[1].detach().cpu(), xyz[2].detach().cpu(), 'co')
            ax.text(xyz[1].detach().cpu(), xyz[2].detach().cpu(), str(ci))
        plt.axis([-0.75, 0.75, -0.75, 0.75])
        plt.draw()
        plt.pause(0.001)

    th = angle.get_angle()
    refined_angles.append(th)

pickle.dump(refined_angles, open('./refined_angles.pkl', 'wb'))