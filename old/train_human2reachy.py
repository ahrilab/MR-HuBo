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
reachy_xyzs, reachy_reps, reachy_angles, smpl_reps, smpl_rots  = split_train_test(reachy_dir, human_dir, num, split_ratio)

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_reachy_angles = 17
dim_smpl_reps = 126
dim_hidden = 256

batch_size = 128
lr = 1e-4
device = 'cuda'

#####################

train_dataset = R4Rdata(reachy_xyzs['train'], reachy_reps['train'], reachy_angles['train'], smpl_reps['train'], smpl_rots['train'])
test_dataset = R4Rdata(reachy_xyzs['test'], reachy_reps['test'], reachy_angles['test'], smpl_reps['test'], smpl_rots['test'])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

####################

# model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_angles+dim_reachy_xyzs+dim_reachy_reps, dim_hidden=dim_hidden).to(device)
model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_angles, dim_hidden=dim_hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-6)

#####################

max_epoch = 50000

best_loss = 100000
criterion = nn.MSELoss()
for epoch in (range(max_epoch)):
    train_loss = 0.0
    model.train()
    for sample in train_dataloader:
        bs = sample['smpl_rep'].shape[0]
        pred = model(sample['smpl_rep'].float().to(device))
        pred_angle = pred[:, :dim_reachy_angles]
        # pred_xyz = pred[:, dim_reachy_angles:dim_reachy_angles+dim_reachy_xyzs]
        # pred_rep = pred[:, dim_reachy_angles+dim_reachy_xyzs:]        
        # pred_rotmat = rotation_6d_to_matrix (pred_rep.reshape(bs, -1, 6))

        # gt_rep = sample['reachy_rep'].float().to(device)
        # gt_rotmat = rotation_6d_to_matrix (gt_rep.reshape(bs, -1, 6))
        # gt_xyz = sample['reachy_xyz'].float().to(device)
        gt_angle = sample['reachy_angle'].float().to(device)
        
        # loss = criterion(pred_xyz, gt_xyz) + get_geodesic_loss(pred_rotmat, gt_rotmat) + criterion(pred_angle, gt_angle)
        loss = criterion(pred_angle, gt_angle)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()/len(train_dataloader)

    test_loss = 0.0
    model.eval()
    for sample in test_dataloader:
        with torch.no_grad():
            bs = sample['smpl_rep'].shape[0]
            pred = model(sample['smpl_rep'].float().to(device))
            pred_angle = pred[:, :dim_reachy_angles]
            # pred_xyz = pred[:, dim_reachy_angles:dim_reachy_angles+dim_reachy_xyzs]
            # pred_rep = pred[:, dim_reachy_angles+dim_reachy_xyzs:]        
            # pred_rotmat = rotation_6d_to_matrix (pred_rep.reshape(bs, -1, 6))

            # gt_rep = sample['reachy_rep'].float().to(device)
            # gt_rotmat = rotation_6d_to_matrix (gt_rep.reshape(bs, -1, 6))
            # gt_xyz = sample['reachy_xyz'].float().to(device)
            gt_angle = sample['reachy_angle'].float().to(device)
            
            # loss = criterion(pred_xyz, gt_xyz) + get_geodesic_loss(pred_rotmat, gt_rotmat) + criterion(pred_angle, gt_angle)
            loss = criterion(pred_angle, gt_angle)
            test_loss += loss.item()/len(test_dataloader)
    
    print('[EPOCH {}] tr loss : {}, te loss :{}'.format(epoch, train_loss, test_loss))

    best_loss = min(best_loss, test_loss)
    if best_loss == test_loss:
        print('save best model')
        torch.save(model.state_dict(), './models/human2reachy_single_best.pth')

