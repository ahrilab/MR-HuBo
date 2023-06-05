from torch.utils.data import DataLoader
from src.data import split_train_test, R4Rdata
from src.net import MLP
import torch.optim as optim
import torch.nn as nn
import torch
from src.loss import get_geodesic_loss
from pytorch3d.transforms import rotation_6d_to_matrix 

num = 500
split_ratio = 50
reachy_dir = './data/reachy/fix'
human_dir = './data/human'
reachy_xyzs, reachy_reps, reachy_angles, smpl_reps, smpl_rots = split_train_test(reachy_dir, human_dir, num, split_ratio, True)

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 512

batch_size = 2048
lr = 1e-4
device = 'cuda'

#####################

train_dataset = R4Rdata(reachy_xyzs['train'], reachy_reps['train'], reachy_angles['train'], smpl_reps['train'], smpl_rots['train'])
test_dataset = R4Rdata(reachy_xyzs['test'], reachy_reps['test'], reachy_angles['test'], smpl_reps['test'], smpl_rots['test'])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

####################

model_pre = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_xyzs+dim_reachy_reps, dim_hidden=dim_hidden).to(device)
model_post = MLP(dim_input=dim_reachy_xyzs+dim_reachy_reps, dim_output=dim_reachy_angles, dim_hidden=dim_hidden).to(device)
model_recon = MLP(dim_input=dim_reachy_xyzs+dim_reachy_reps, dim_output=dim_smpl_reps, dim_hidden=dim_hidden).to(device)

optimizer_pre = optim.Adam(model_pre.parameters(), lr, weight_decay=1e-6)
optimizer_post = optim.Adam(model_post.parameters(), lr, weight_decay=1e-6)
optimizer_recon = optim.Adam(model_recon.parameters(), lr, weight_decay=1e-6)

#####################

max_epoch = 1000

best_pre_loss = 100000
best_post_loss = 100000
best_recon_loss = 100000
criterion = nn.MSELoss()
for epoch in (range(max_epoch)):
    train_pre_loss = 0.0
    train_post_loss = 0.0
    train_recon_loss = 0.0
    model_pre.train()
    model_post.train()
    model_recon.train()
    for sample in train_dataloader:
        bs = sample['smpl_rep'].shape[0]
        gt_smpl_rep = sample['smpl_rep'].float().to(device)
        pre_reachy_pred = model_pre(gt_smpl_rep)
        
        pred_reachy_xyz = pre_reachy_pred[:, :dim_reachy_xyzs]
        pred_reachy_rep = pre_reachy_pred[:, dim_reachy_xyzs:]        
        pred_reachy_rotmat = rotation_6d_to_matrix(pred_reachy_rep.reshape(bs, -1, 6))

        gt_reachy_rep = sample['reachy_rep'].float().to(device)
        gt_reachy_rotmat = rotation_6d_to_matrix(gt_reachy_rep.reshape(bs, -1, 6))
        gt_reachy_xyz = sample['reachy_xyz'].float().to(device)
        gt_reachy_angle = sample['reachy_angle'].float().to(device)

        post_gt_inp = torch.cat([gt_reachy_xyz, gt_reachy_rep], dim=-1)
        post_pred_inp = torch.cat([pred_reachy_xyz, pred_reachy_rep], dim=-1)
        
        post_pred_teacher = model_post(post_gt_inp)
        pred_angle_teacher = post_pred_teacher[:, :dim_reachy_angles]
        
        post_pred_student = model_post(post_pred_inp.detach())
        pred_angle_student = post_pred_student[:, :dim_reachy_angles]

        smpl_pred_teacher = model_recon(post_gt_inp)
        smpl_pred_student = model_recon(post_pred_inp.detach())

        pre_loss = criterion(pred_reachy_xyz, gt_reachy_xyz) + criterion(pred_reachy_rep, gt_reachy_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
        post_loss = criterion(pred_angle_teacher, gt_reachy_angle) + criterion(pred_angle_student, gt_reachy_angle)
        recon_loss = criterion(smpl_pred_teacher, gt_smpl_rep) + criterion(smpl_pred_student, gt_smpl_rep)

        optimizer_pre.zero_grad()
        pre_loss.backward()
        optimizer_pre.step()

        optimizer_post.zero_grad()
        post_loss.backward()
        optimizer_post.step()

        optimizer_recon.zero_grad()
        recon_loss.backward()
        optimizer_recon.step()

        train_pre_loss += pre_loss.item()/len(train_dataloader)
        train_post_loss += post_loss.item()/len(train_dataloader)
        train_recon_loss += recon_loss.item()/len(train_dataloader)

    test_pre_loss = 0.0
    test_post_loss = 0.0
    test_recon_loss = 0.0
    model_pre.eval()
    model_post.eval()
    model_recon.eval()
    for sample in test_dataloader:
        with torch.no_grad():
            bs = sample['smpl_rep'].shape[0]
            gt_smpl_rep = sample['smpl_rep'].float().to(device)
            pre_reachy_pred = model_pre(gt_smpl_rep)
            
            pred_reachy_xyz = pre_reachy_pred[:, :dim_reachy_xyzs]
            pred_reachy_rep = pre_reachy_pred[:, dim_reachy_xyzs:]        
            pred_reachy_rotmat = rotation_6d_to_matrix(pred_reachy_rep.reshape(bs, -1, 6))

            gt_reachy_rep = sample['reachy_rep'].float().to(device)
            gt_reachy_rotmat = rotation_6d_to_matrix(gt_reachy_rep.reshape(bs, -1, 6))
            gt_reachy_xyz = sample['reachy_xyz'].float().to(device)
            gt_reachy_angle = sample['reachy_angle'].float().to(device)

            post_gt_inp = torch.cat([gt_reachy_xyz, gt_reachy_rep], dim=-1)
            post_pred_inp = torch.cat([pred_reachy_xyz, pred_reachy_rep], dim=-1)
            
            post_pred_teacher = model_post(post_gt_inp)
            pred_angle_teacher = post_pred_teacher[:, :dim_reachy_angles]
            
            post_pred_student = model_post(post_pred_inp.detach())
            pred_angle_student = post_pred_student[:, :dim_reachy_angles]

            smpl_pred_teacher = model_recon(post_gt_inp)
            smpl_pred_student = model_recon(post_pred_inp.detach())

            pre_loss = criterion(pred_reachy_xyz, gt_reachy_xyz) + criterion(pred_reachy_rep, gt_reachy_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
            post_loss = criterion(pred_angle_teacher, gt_reachy_angle) + criterion(pred_angle_student, gt_reachy_angle)
            recon_loss = criterion(smpl_pred_teacher, gt_smpl_rep) + criterion(smpl_pred_student, gt_smpl_rep)

            test_pre_loss += pre_loss.item()/len(test_dataloader)
            test_post_loss += post_loss.item()/len(test_dataloader)
            test_recon_loss += recon_loss.item()/len(test_dataloader)
    
    print('[EPOCH {}] tr loss : {:.03f},{:.03f}, {:.03f} te loss :{:.03f},{:.03f}, {:.03f}'.format(epoch, train_pre_loss, train_post_loss, train_recon_loss,
                                                                                                      test_pre_loss, test_post_loss, test_recon_loss))

    best_pre_loss = min(best_pre_loss, test_pre_loss)
    if best_pre_loss == test_pre_loss:
        print('save best pre model')
        torch.save(model_pre.state_dict(), './models/human2reachy_best_pre_v3.pth')

    best_post_loss = min(best_post_loss, test_post_loss)
    if best_post_loss == test_post_loss:
        print('save best post model')
        torch.save(model_post.state_dict(), './models/human2reachy_best_post_v3.pth')

    best_recon_loss = min(best_recon_loss, test_recon_loss)
    if best_recon_loss == test_recon_loss:
        print('save best recon model')
        torch.save(model_recon.state_dict(), './models/human2reachy_best_recon_v3.pth')
