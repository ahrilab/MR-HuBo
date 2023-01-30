from torch.utils.data import DataLoader
from src.data import split_train_test, R4Rdata
from src.net import MLP
import torch.optim as optim
import torch.nn as nn
import torch
from src.loss import rep2rotmat, get_geodesic_loss

num = 100
split_ratio = 20
reachy_dir = './data/reachy/raw'
human_dir = './data/human'
reachy_xyzs, reachy_reps, smpl_reps, smpl_rots  = split_train_test(reachy_dir, human_dir, num, split_ratio)

dim_reachy_xyzs = 93
dim_reachy_reps = 186
dim_smpl_reps = 126
dim_hidden = 512

batch_size = 64
lr = 1e-4
device = 'cuda'

#####################

train_dataset = R4Rdata(reachy_xyzs['train'], reachy_reps['train'], smpl_reps['train'], smpl_rots['train'])
test_dataset = R4Rdata(reachy_xyzs['test'], reachy_reps['test'], smpl_reps['test'], smpl_rots['test'])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

####################

model = MLP(dim_input=dim_smpl_reps, dim_output=dim_reachy_reps, dim_hidden=dim_hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr)

#####################

max_epoch = 1000

best_loss = 1000
for epoch in (range(max_epoch)):
    train_loss = 0.0
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for sample in train_dataloader:
        bs = sample['smpl_rep'].shape[0]
        pred = model(sample['smpl_rep'].float().to(device))
        pred_rotmat = rep2rotmat(pred.reshape(bs, -1, 3, 2))

        gt = sample['reachy_rep'].float().to(device)
        gt_rotmat = rep2rotmat(gt.reshape(bs, -1, 3, 2))

        loss = get_geodesic_loss(pred_rotmat, gt_rotmat)

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
            pred_rotmat = rep2rotmat(pred.reshape(bs, -1, 3, 2))

            gt = sample['reachy_rep'].float().to(device)
            gt_rotmat = rep2rotmat(gt.reshape(bs, -1, 3, 2))

            loss = get_geodesic_loss(pred_rotmat, gt_rotmat)

        test_loss += loss.item()/len(test_dataloader)
    
    print('[EPOCH {}] tr loss : {}, te loss :{}'.format(epoch, train_loss, test_loss))

    best_loss = min(best_loss, test_loss)
    if best_loss == test_loss:
        print('save best model')
        torch.save(model.state_dict(), './models/human2reachy_best.pth')

