import kinpy as kp
import numpy as np
from kinpy.transform import Transform
from src.net import MLP
import joblib
import torch
import pickle
from src.loss import rep2rotmat
from src.misc import joint_range, ret_keys
from src.viz import render_ret
from src.hbp import transform_smpl_coordinate
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import matrot2aa, aa2matrot

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 256

device = 'cuda'

chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())
# serial_chain = kp.build_serial_chain_from_urdf(open('./reachy.urdf').read(), 'right_tip')
#####################

model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_angles, dim_hidden=dim_hidden).to(device)
# model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_angles+dim_reachy_xyzs+dim_reachy_reps, dim_hidden=dim_hidden).to(device)
model.load_state_dict(torch.load('./models/human2reachy_best.pth'))
model.eval()

#####################
num_betas = 16
result = joblib.load(open('./pymaf.pkl', 'rb'))

# smpl_data = {}
# smpl_data['betas'] = np.zeros((num_betas, ))
# smpl_data['pose_body'] = result['pose'][:, 3:66]
# smpl_data['root_orient'] = np.zeros((len(result['pose']), 3))
# smpl_data['trans'] = np.zeros((len(result['pose']), 3))
# smpl_data['trans'][:, 1] += 0.5

# transformed_d = transform_smpl_coordinate(bm_fname='./data/bodymodel/smplx/neutral.npz', 
#                                         trans=smpl_data['trans'], 
#                                         root_orient=smpl_data['root_orient'],
#                                         betas=smpl_data['betas'], 
#                                         rotxyz=[90, 0, 0])
# smpl_data.update(transformed_d)

# smpl_data['surface_model_type'] = 'smplx'
# smpl_data['gender'] = 'neutral'
# smpl_data['mocap_frame_rate'] = 30
# smpl_data['num_betas'] = num_betas

vid_pose = result['pose'][:, 3:66]

joint_keys = sorted([k for k, v in joint_range.items()])

length = len(vid_pose)
smpl_aa = vid_pose.reshape(length, -1, 3)
num_joints = smpl_aa.shape[1]
smpl_aa = smpl_aa.reshape(length * num_joints, 3)

smpl_rot = aa2matrot(torch.from_numpy(smpl_aa))
smpl_rep = matrix_to_rotation_6d(smpl_rot)

smpl_rep = smpl_rep.reshape(length, num_joints, 6).reshape(length, -1)

with torch.no_grad():
    pred = model(smpl_rep.to(device).float())
    pred = pred.detach().cpu().numpy()[:, :dim_reachy_angles]

reachy_angles = []
for p in pred:
    reachy_angles.append({k: p[i] for i, k in enumerate(joint_keys)})

pickle.dump(reachy_angles, open('./pymaf_robot.pkl', 'wb'))
print('Finish!')
# with torch.no_grad():
#     reachy_data = model(smpl_rep.to(device).float())
#     reachy_xyz = reachy_data[:, dim_reachy_angles:dim_reachy_angles+dim_reachy_xyzs].reshape(length, -1, 3)
#     reachy_rep = reachy_data[:, dim_reachy_angles+dim_reachy_xyzs:]
#     reachy_rotmat = rotation_6d_to_matrix(reachy_rep.reshape(length, -1, 6))
#     reachy_quat = matrix_to_quaternion(reachy_rotmat)

# reachy_xyz = reachy_xyz.detach().cpu().numpy()
# reachy_quat = reachy_quat.detach().cpu().numpy()

# for t in range(length):
#     curr_xyz = reachy_xyz[t]
#     curr_rot = reachy_quat[t]

#     ret = dict()

#     for ki, k in enumerate(ret_keys):
#         ret[k] = Transform(rot=curr_rot[ki], pos=curr_xyz[ki])

#     render_ret(ret, chain, './test.png', 1280)
#     print()
