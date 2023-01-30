import kinpy as kp
from kinpy.transform import Transform
from src.net import MLP
import joblib
import torch
import pickle
from src.loss import rep2rotmat
from src.misc import joint_range, ret_keys
from src.viz import render_ret
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 512

device = 'cuda'

chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())
# serial_chain = kp.build_serial_chain_from_urdf(open('./reachy.urdf').read(), 'right_tip')
#####################

model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_angles, dim_hidden=dim_hidden).to(device)
model.load_state_dict(torch.load('./models/human2reachy_best.pth'))
model.eval()

#####################

vid_out = joblib.load(open('./pymaf.pkl', 'rb'))
vid_pose = vid_out['pose'][:, 3:66]

joint_keys = sorted([k for k, v in joint_range.items()])

length = len(vid_pose)
smpl_aa = vid_pose.reshape(length, -1, 3)

smpl_rot = axis_angle_to_matrix(torch.from_numpy(smpl_aa))
smpl_rep = matrix_to_rotation_6d(smpl_rot).reshape(length, -1)

with torch.no_grad():
    pred = model(smpl_rep.to(device).float())
    pred = pred.detach().cpu().numpy()

reachy_angles = []
for p in pred:
    reachy_angles.append({k: p[i] for i, k in enumerate(joint_keys)})

pickle.dump(reachy_angles, open('./pymaf_robot.pkl', 'wb'))
print('Finish!')
# with torch.no_grad():
#     reachy_data = model(smpl_rep.to(device).float())
#     reachy_xyz = reachy_data[:, :dim_reachy_xyzs].reshape(length, -1, 3)
#     reachy_rep = reachy_data[:, dim_reachy_xyzs:]
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

#     print(ret)

