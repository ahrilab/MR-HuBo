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

#############################################
def sample(num_poses, seed=None):
    np.random.seed(seed)
    with torch.no_grad():
        Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, 32)), dtype=torch.float32, device=device)
    return vp.decode(Zgen)

def add_noise_to_pose(aa, num_poses, seed=None):
    np.random.seed(seed)
    with torch.no_grad():
        Z_main = vp.encode(aa).mean
        noise = torch.tensor(np.random.normal(0., 0.5, size=(num_poses-1, 32)), dtype=torch.float32, device=device)
        Z_noisy = Z_main + noise
        Zgen = torch.cat([Z_main, Z_noisy], dim=0)
    return vp.decode(Zgen)['pose_body']

###########################################
num_betas = 16
num_joints =21
num_samples = 1000
num_plots = 9

pymaf = joblib.load(open('./pymaf.pkl', 'rb'))
idx = 350

smpl_aa = pymaf['pose'][:, 3:66][idx:idx+1]

# joint_keys = sorted([k for k, v in joint_range.items()])
# length = len(smpl_aa)
# smpl_aa = smpl_aa.reshape(length, -1, 3)
# num_joints = smpl_aa.shape[1]
# smpl_aa = smpl_aa.reshape(length * num_joints, 3)
smpl_aa = torch.from_numpy(smpl_aa)

smpl_plot_noisy_aa = add_noise_to_pose(smpl_aa.to(device), num_plots, seed=0).contiguous().view(num_plots, -1)
smpl_stat_noisy_aa = add_noise_to_pose(smpl_aa.to(device), num_samples, seed=0)
smpl_stat_noisy_rot = aa2matrot(smpl_stat_noisy_aa.reshape(num_samples*num_joints, -1))
smpl_stat_noisy_rep = matrix_to_rotation_6d(smpl_stat_noisy_rot)
smpl_stat_noisy_rep = smpl_stat_noisy_rep.reshape(num_samples, num_joints, 6).reshape(num_samples, -1)

with torch.no_grad():
    reachy_stat_noisy_xyzrep = model_pre(smpl_stat_noisy_rep)
    reachy_stat_noisy_xyz = reachy_stat_noisy_xyzrep[:, :dim_reachy_xyzs]
    reachy_stat_noisy_rep = reachy_stat_noisy_xyzrep[:, dim_reachy_xyzs:]
    reachy_stat_noisy_angle = model_post(reachy_stat_noisy_xyzrep)[:, :dim_reachy_angles]

reachy_angle_var = torch.var(reachy_stat_noisy_angle, dim=0)
reachy_xyz_var = torch.var(reachy_stat_noisy_xyz.reshape(num_samples, -1, 3), dim=[0, 2])
for ki, k in enumerate(joint_range.keys()):
    print(k, '\t', reachy_angle_var[ki].item())
print('='*50)
for ki, k, in enumerate(ret_keys):
    print(k, '\t', reachy_xyz_var[ki].item())

reachy_dist_mat = torch.cdist(reachy_stat_noisy_xyz.reshape(num_samples, -1, 3), reachy_stat_noisy_xyz.reshape(num_samples, -1, 3))
reachy_dist_mat = reachy_dist_mat / torch.sum(reachy_dist_mat, dim=2)[:, :, None]
reachy_dist_std = torch.std(reachy_dist_mat, dim=0)
reachy_dist_mean = torch.mean(reachy_dist_mat, dim=0)
# plt.imshow(reachy_dist_std.detach().cpu()[1:, 1:])
# plt.show()

bm_res = bm(**{'pose_body':smpl_stat_noisy_aa.reshape(num_samples, -1).to(device)})
smpl_xyz = bm_res.Jtr[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,45,23,24]] # todo : add left right...
smpl_dist_mat = torch.cdist(smpl_xyz, smpl_xyz)
smpl_dist_mat = smpl_dist_mat / torch.sum(smpl_dist_mat, dim=2)[:, :, None]
smpl_dist_std = torch.std(smpl_dist_mat, dim=0)
smpl_dist_mean = torch.mean(smpl_dist_mat, dim=0)
plt.imshow(smpl_dist_std.detach().cpu())
plt.show()
###################################################w

smpl_ori_dist_mat = torch.cdist(smpl_xyz[0], smpl_xyz[0])
smpl_ori_dist_mat = smpl_ori_dist_mat / torch.sum(smpl_ori_dist_mat, dim=-1)[:, None]
plt.imshow(smpl_ori_dist_mat.detach().cpu())
plt.show()

###################################################
smpl_cidx = [14,15,16,17,18,19,20]#, 19, 20]#,21,22]#,23,24] # 11 - 14
gt_dist_mat = torch.cdist(smpl_xyz[0][smpl_cidx], smpl_xyz[0][smpl_cidx])
gt_dist_mat = gt_dist_mat / torch.sum(gt_dist_mat, dim=-1)[:, None]
plt.imshow(gt_dist_mat.detach().cpu())
plt.show()

roa = reachy_stat_noisy_angle[0].detach().cpu()
# roa = [roa[0], roa[1], roa[2], roa[3], roa[4], roa[5], roa[6], 0.0, 
#        roa[7], roa[8], roa[9], roa[10],roa[11],roa[12], roa[13], 0.0, 
#        roa[14], roa[15], roa[16], 0.0, 0.0 ]
# roa = torch.Tensor(roa)

class Angle(torch.nn.Module):
    def __init__(self, roa, chain):
        super().__init__()
        self.target = torch.nn.Parameter(roa, requires_grad=True)
        self.chain = chain

    def forward(self):
        th = {}
        for ki, k in enumerate(sorted(joint_range.keys())):
            th[k] = self.target[ki]
        
        fk_res = self.chain.forward_kinematics(th)
        fk_rotmat = []
        fk_xyz = []
        
        ## 
        for k in ['r_shoulder', 'r_upper_arm', 'r_wrist2hand', 'l_shoulder', 'l_upper_arm', 'l_wrist2hand', 'head']:#, 'l_wrist2hand', 'r_wrist2hand',]:# 'left_camera', 'right_camera']:
            #           0            1            2          3             4             5         6
            fk_rotmat.append(fk_res[k].get_matrix()[:, :3, :3])
            fk_xyz.append(fk_res[k].get_matrix()[:, :-1, -1])
        fk_rep = matrix_to_rotation_6d(torch.cat(fk_rotmat))
        fk_xyz = torch.cat(fk_xyz)
        spine_1 = torch.Tensor([0.0, 0.0, 0.6])
        spine_2 = torch.Tensor([0.0, 0.0, 0.7])
        spine_3 = torch.Tensor([0.0, 0.0, 0.8])
        left_hip = torch.Tensor([0.0, 0.15, 0.7])
        right_hip = torch.Tensor([0.0, -0.15, 0.7])
                     #      0        1       2        3          4      5         6            7        8             9          10        11
        fk_xyz4smpl = [fk_xyz[6], fk_xyz[3], fk_xyz[0], fk_xyz[4], fk_xyz[1], fk_xyz[5], fk_xyz[2]]#, fk_xyz[7], fk_xyz[8],]#fk_xyz[9], fk_xyz[10]]
        # fk_xyz4smpl = [left_hip, right_hip, spine_1, spine_2, spine_3, fk_xyz[6], fk_xyz[3], fk_xyz[0], fk_xyz[4], fk_xyz[1], fk_xyz[5], fk_xyz[2]]#, fk_xyz[7], fk_xyz[8],]#fk_xyz[9], fk_xyz[10]]
        # fk_xyz4smpl = [spine_1, spine_2, spine_3, fk_xyz[6], fk_xyz[3], fk_xyz[0], fk_xyz[4], fk_xyz[1], fk_xyz[5], fk_xyz[2], fk_xyz[7], fk_xyz[8],]#fk_xyz[9], fk_xyz[10]]
        # fk_xyz4smpl = [fk_xyz[6], fk_xyz[3], fk_xyz[0], fk_xyz[4], fk_xyz[1], fk_xyz[5], fk_xyz[2]]#, fk_xyz[7], fk_xyz[8],]#fk_xyz[9], fk_xyz[10]]
        fk_xyz4smpl = torch.cat([f[None] for f in fk_xyz4smpl])
        fk_dist_mat = torch.cdist(fk_xyz4smpl, fk_xyz4smpl)
        fk_dist_mat = fk_dist_mat / torch.sum(fk_dist_mat, dim=-1)[:, None]

        return fk_dist_mat

#####################################################
smpl_large_aa = sample(num_samples, seed=0)['pose_body']
smpl_mean = torch.mean(smpl_large_aa, dim=[0, 2])
smpl_std = torch.std(smpl_large_aa, dim=[0, 2])
smpl_stat_noisy_aa = (smpl_stat_noisy_aa - smpl_mean[None, :, None])/smpl_std[None, :, None]

joint_var = torch.var(smpl_stat_noisy_aa, dim=[0, 2]).detach().cpu().numpy()
print('='*50)
for ij, jn in enumerate(smplx_jname):
    print(ij, jn, '\t {:.05f}'.format(joint_var[ij].item(),))
loss_weight = np.array([joint_var[14], joint_var[15], joint_var[16], joint_var[17], joint_var[18], joint_var[19], joint_var[20]])#, joint_var[19], joint_var[20], ] )

chain = pk.build_chain_from_urdf(open('./reachy.urdf').read())
angle = Angle(roa, chain)

optimizer = torch.optim.SGD(angle.parameters(), lr=5)

process = []
for i in range(100):
    tgt_dict = dict()
    for ki, k in enumerate(sorted(joint_range.keys())):
        tgt_dict[k] = copy.deepcopy(angle.target).detach().cpu().numpy()[ki]
    
    if i % 20 == 0:
        process.append(tgt_dict)
        
    fk_dist_mat = angle()
    loss = (fk_dist_mat - gt_dist_mat.detach().cpu()) ** 2
    loss = torch.Tensor(loss_weight)[:, None] * loss
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # )

pickle.dump(process, open('./process.pkl', 'wb'))
# plt.imshow(fk_dist_mat.detach().cpu())
# plt.show()

## plot
smpl_dict = np.load(smpl_path)
mean_pose_hand = np.repeat(np.concatenate([smpl_dict['hands_meanl'], smpl_dict['hands_meanr']])[None], axis=0, repeats=num_plots)
mean_pose_hand = torch.from_numpy(mean_pose_hand)

images = render_smpl_params(bm, {'pose_body': smpl_plot_noisy_aa.to(device),
                                 'pose_hand': mean_pose_hand.to(device)}).reshape(3,3,1,800,800,3)
img = imagearray2file(images)
show_image(np.array(img[0]))


plt.show()

print()