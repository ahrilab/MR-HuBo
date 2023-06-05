import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.distributions import multivariate_normal
from src.misc import smplx_jname
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import matrot2aa, aa2matrot

from sklearn.decomposition import PCA
from scipy.stats import normaltest
import random

device = 'cuda'

vposer_dir = './data/vposer_v2_05'
smpl_path = './data/bodymodel/smplx/neutral.npz'

###########################################
bm = BodyModel(bm_fname=smpl_path).to(device)

###########################################
vp, ps = load_model(vposer_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
vp = vp.to(device)

##########################################
def draw(probs):
    val = random.random()
    csum = 0
    for i, p in enumerate(probs):
        csum  += p
        if csum > val:
            return i
##########################################
num_poses = 25 # number of body poses in each batch

def sample(num_poses, seed=None):
    np.random.seed(seed)
    with torch.no_grad():
        Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, 32)), dtype=torch.float32, device=device)
    return vp.decode(Zgen)

def add_noise_to_z(num_poses, seed=None):
    np.random.seed(seed)
    with torch.no_grad():
        Z_main = torch.tensor(np.random.normal(0., 1., size=(1, 32)), dtype=torch.float32, device=device)
        noise = torch.tensor(np.random.normal(0., 0.2, size=(num_poses-1, 32)), dtype=torch.float32, device=device)
        Z_noisy = Z_main + noise
        Zgen = torch.cat([Z_main, Z_noisy], dim=0)
    return vp.decode(Zgen)

def add_noise_to_joint(num_trials, seed=None):
    np.random.seed(seed)
    with torch.no_grad():
        Zgen = torch.tensor(np.random.normal(0., 1., size=(1, 32)), dtype=torch.float32, device=device)
    gen_joint = vp.decode(Zgen)['pose_body'] # 1 21 3

    joint_dist = []
    joint_samples = []

    for ji in range(gen_joint.shape[1]):
        dist_list = []
        for _ in range(num_trials):
            copy_joint = copy.deepcopy(gen_joint)
            joint_noise = torch.tensor(np.random.normal(0., 0.1, size=(1, 3)), dtype=torch.float32, device=device)
            copy_joint[:, ji, :] += joint_noise
            copy_z = vp.encode(copy_joint.reshape(1, -1)).mean

            z_dist = torch.norm(Zgen-copy_z)# torch.cosine_similarity(Zgen, copy_z).item() #torch.norm(Zgen-copy_z)

            dist_list.append(z_dist)
        dist_mean = torch.mean(torch.Tensor(dist_list))
        joint_dist.append(dist_mean.item())
        joint_samples.append(copy_joint)
    return gen_joint, joint_samples, joint_dist

def show_image(img_ndarray):
    '''
    Visualize rendered body images resulted from render_smpl_params in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

# seed 0, 1, 990, 10011 999 73 39 42 45 51 155 229 2293 193 143 1243 743 83 81 315
# 222
num_joints = 21
num_seed = 1000
num_samples = 3000
list_joint_var = []

# gen_joint, joint_samples, joint_dist = add_noise_to_joint(num_samples, seed=seed)

# plot_joints = torch.cat([gen_joint, gen_joint, gen_joint, gen_joint, torch.cat(joint_samples)])
# for ij, jn in enumerate(smplx_jname):
#     print(jn, '\t {:.05f}'.format(joint_dist[ij]))

# images = render_smpl_params(bm, {'pose_body':plot_joints.reshape(25, -1)}).reshape(5,5,1,800,800,3)
# img = imagearray2file(images)
# show_image(np.array(img[0]))
# plt.savefig('test.png')

# for seed in range(num_seed):
#     more_pose_body = add_noise_to_z(num_samples, seed=seed)['pose_body']
#     joint_var = torch.var(more_pose_body, dim=[0, 2])
#     list_joint_var.append(joint_var[None, :])
# std_of_joint_var = torch.std(torch.cat(list_joint_var), dim=0)
# mean_of_joint_var = torch.mean(torch.cat(list_joint_var), dim=0)

pose_samples = sample(num_samples, seed=0)['pose_body']
# joint_var = torch.var(pose_samples, dim=[0, 2])
#     list_joint_var.append(joint_var[None, :])
std_of_joint = torch.std(pose_samples, dim=[0, 2])
mean_of_joint = torch.mean(pose_samples, dim=[0, 2])

seed = 143

sampled_pose_body = add_noise_to_z(num_poses, seed=seed)['pose_body']
sampled_pose_body = sampled_pose_body.contiguous().view(num_poses, -1) # will a generate Nx1x21x3 tensor of body poses 
more_pose_body = add_noise_to_z(num_samples, seed=seed)['pose_body']
more_pose_body = (more_pose_body - mean_of_joint[None, :, None])/std_of_joint[None, :, None]
# variance of xyz position????
joint_var = torch.var(more_pose_body, dim=[0, 2])
# cos_sim = torch.cosine_similarity(more_pose_body[0:1].reshape(1, -1), 
#                                    more_pose_body[1:].reshape(num_samples-1, -1), dim=-1)
# print(torch.mean(cos_sim))
# joint_var = (joint_var - mean_of_joint)/(std_of_joint)

# body_param = {'pose_body':more_pose_body.reshape(num_samples, -1)}
# bm_res = bm(**body_param)
# jtr = bm_res.Jtr[:, 1:22]
# jtr_var = torch.var(jtr, dim=[0, 2])
images = render_smpl_params(bm, {'pose_body':sampled_pose_body}).reshape(5,5,1,800,800,3)
img = imagearray2file(images)
show_image(np.array(img[0]))
plt.savefig('test.png')
# plt.show()

# plt.plot(jtr[0, :, 0].detach().cpu().numpy(), jtr[0, :, 1].detach().cpu().numpy(), 'o')
# for ij, j in enumerate(jtr[0]):
#     plt.text(j[0], j[1], str(ij))

for ij, jn in enumerate(smplx_jname):
    print(jn, '\t {:.05f}'.format(joint_var[ij].item(),))
print(torch.std(more_pose_body))
