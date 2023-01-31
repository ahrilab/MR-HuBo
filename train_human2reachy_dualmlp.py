from torch.utils.data import DataLoader
from src.data import split_train_test, R4Rdata
from src.net import MLP
import torch.optim as optim
import torch.nn as nn
import torch
from src.loss import get_geodesic_loss
from pytorch3d.transforms import rotation_6d_to_matrix 

num = 20
split_ratio = 10
reachy_dir = './data/reachy/fix'
human_dir = './data/human'
reachy_xyzs, reachy_reps, reachy_angles, smpl_reps, smpl_rots  = split_train_test(reachy_dir, human_dir, num, split_ratio)

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 512

batch_size = 64
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
optimizer_pre = optim.Adam(model_pre.parameters(), lr, weight_decay=1e-6)
optimizer_post = optim.Adam(model_post.parameters(), lr, weight_decay=1e-6)

#####################

max_epoch = 100

best_pre_loss = 100000
best_post_loss = 100000
criterion = nn.MSELoss()
for epoch in (range(max_epoch)):
    train_pre_loss = 0.0
    train_post_loss = 0.0
    model_pre.train()
    model_post.train()
    for sample in train_dataloader:
        bs = sample['smpl_rep'].shape[0]
        pre_pred = model_pre(sample['smpl_rep'].float().to(device))
        # pred_angle = pred[:, :dim_reachy_angles]
        pred_xyz = pre_pred[:, :dim_reachy_xyzs]
        pred_rep = pre_pred[:, dim_reachy_xyzs:]        
        pred_rotmat = rotation_6d_to_matrix(pred_rep.reshape(bs, -1, 6))

        gt_rep = sample['reachy_rep'].float().to(device)
        gt_rotmat = rotation_6d_to_matrix(gt_rep.reshape(bs, -1, 6))
        gt_xyz = sample['reachy_xyz'].float().to(device)
        gt_angle = sample['reachy_angle'].float().to(device)

        post_inp = torch.cat([gt_xyz, gt_rep], dim=-1)
        post_pred = model_post(post_inp)
        pred_angle = post_pred[:, :dim_reachy_angles]

        pre_loss = criterion(pred_xyz, gt_xyz) + criterion(pred_rep, gt_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
        post_loss = criterion(pred_angle, gt_angle)
        
        optimizer_pre.zero_grad()
        pre_loss.backward()
        optimizer_pre.step()

        optimizer_post.zero_grad()
        post_loss.backward()
        optimizer_post.step()

        train_pre_loss += pre_loss.item()/len(train_dataloader)
        train_post_loss += post_loss.item()/len(train_dataloader)

    test_pre_loss = 0.0
    test_post_loss = 0.0
    model_pre.eval()
    model_post.eval()
    for sample in test_dataloader:
        with torch.no_grad():
            bs = sample['smpl_rep'].shape[0]
            pre_pred = model_pre(sample['smpl_rep'].float().to(device))
            pred_xyz = pre_pred[:, :dim_reachy_xyzs]
            pred_rep = pre_pred[:, dim_reachy_xyzs:]        
            pred_rotmat = rotation_6d_to_matrix(pred_rep.reshape(bs, -1, 6))

            gt_rep = sample['reachy_rep'].float().to(device)
            gt_rotmat = rotation_6d_to_matrix(gt_rep.reshape(bs, -1, 6))
            gt_xyz = sample['reachy_xyz'].float().to(device)
            gt_angle = sample['reachy_angle'].float().to(device)

            post_inp = torch.cat([gt_xyz, gt_rep], dim=-1)
            post_pred = model_post(post_inp)
            pred_angle = post_pred[:, :dim_reachy_angles]

            pre_loss = criterion(pred_xyz, gt_xyz) + criterion(pred_rep, gt_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
            post_loss = criterion(pred_angle, gt_angle)

            test_pre_loss += pre_loss.item()/len(test_dataloader)
            test_post_loss += post_loss.item()/len(test_dataloader)
    
    print('[EPOCH {}] tr loss : {:.03f},{:.03f} te loss :{:.03f},{:.03f}'.format(epoch, train_pre_loss, train_post_loss, 
                                                                                 test_pre_loss, test_post_loss))

    best_pre_loss = min(best_pre_loss, test_pre_loss)
    if best_pre_loss == test_pre_loss:
        print('save best pre model')
        torch.save(model_pre.state_dict(), './models/human2reachy_best_pre.pth')

    best_post_loss = min(best_post_loss, test_post_loss)
    if best_post_loss == test_post_loss:
        print('save best post model')
        torch.save(model_post.state_dict(), './models/human2reachy_best_post.pth')
