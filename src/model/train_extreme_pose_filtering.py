import argparse
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch3d.transforms import rotation_6d_to_matrix

sys.path.append("./src")
# from utils.loss import get_geodesic_loss
from utils.data_with_prob import split_train_test, H2RMotionData

from model.net import MLP
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TrainArgs
from utils.consts import *

from torch.distributions.bernoulli import Bernoulli


def train(args: TrainArgs):
    robot_config = RobotConfig(args.robot_type)

    num_data = NUM_SEEDS
    split_ratio = 50  # 2

    # fmt: off
    input_path = robot_config.ROBOT_TO_SMPL_PATH    # input: SMPL parameters
    target_path = robot_config.RAW_DATA_PATH        # target: robot joint angles
    # fmt: on

    robot_xyzs, robot_reps, robot_angles, smpl_reps, smpl_rots, smpl_prob = split_train_test(
        input_path, target_path, num_data, split_ratio, False
    )

    dim_hidden = HIDDEN_DIM
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    device = DEVICE

    #####################

    train_dataset = H2RMotionData(
        robot_xyzs["train"],
        robot_reps["train"],
        robot_angles["train"],
        smpl_reps["train"],
        smpl_rots["train"],
        smpl_prob["train"],
    )
    test_dataset = H2RMotionData(
        robot_xyzs["test"],
        robot_reps["test"],
        robot_angles["test"],
        smpl_reps["test"],
        smpl_rots["test"],
        smpl_prob["test"],
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    ####################

    model_pre = MLP(
        dim_input=robot_config.smpl_reps_dim,
        dim_output=robot_config.xyzs_dim + robot_config.reps_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    model_post = MLP(
        dim_input=robot_config.xyzs_dim + robot_config.reps_dim,
        dim_output=robot_config.angles_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    optimizer_pre = optim.Adam(model_pre.parameters(), lr, weight_decay=1e-6)
    optimizer_post = optim.Adam(model_post.parameters(), lr, weight_decay=1e-6)

    #####################
    best_pre_loss = 100000
    best_post_loss = 100000
    criterion = nn.MSELoss()
    for epoch in range(NUM_EPOCHS):
        train_pre_loss = 0.0
        train_post_loss = 0.0
        model_pre.train()
        model_post.train()
        for sample in train_dataloader:
            ### Add: for each iteration probability
            prob = sample["smpl_prob"]
            bernoulli_dist = Bernoulli(prob)

            # Sample from each Bernoulli distribution
            chosen_samples = bernoulli_dist.sample()
            # print(len(chosen_samples))

            # we need to find the number alpha
            alpha = 1200
            sample_index = (chosen_samples == 1).nonzero()[:alpha, 0]  # we use 100 to set the max number in batch

            ### We still need to do this for each sample[" "]
            # Extract the corresponding values in random_samples

            filtered_smpl_rep = sample["smpl_rep"][sample_index]

            bs = filtered_smpl_rep.shape[0]
            print("Number of T per batch: ", len((chosen_samples == 1).nonzero()[:, 0]), sample["smpl_prob"].shape[0])
            print("BATCH SIZE: ", bs)

            pre_pred = model_pre(filtered_smpl_rep.float().to(device))

            # pred_angle = pred[:, :robot_config.angles_dim]
            pred_xyz = pre_pred[:, : robot_config.xyzs_dim]
            pred_rep = pre_pred[:, robot_config.xyzs_dim :]
            # pred_rotmat = rotation_6d_to_matrix(pred_rep.reshape(bs, -1, 6))

            gt_rep = sample["robot_rep"][sample_index].float().to(device)
            # gt_rotmat = rotation_6d_to_matrix(gt_rep.reshape(bs, -1, 6))
            gt_xyz = sample["robot_xyz"][sample_index].float().to(device)
            gt_angle = sample["robot_angle"][sample_index].float().to(device)

            post_gt_inp = torch.cat([gt_xyz, gt_rep], dim=-1)
            post_pred_inp = torch.cat([pred_xyz, pred_rep], dim=-1)

            post_pred_teacher = model_post(post_gt_inp)
            pred_angle_teacher = post_pred_teacher[:, : robot_config.angles_dim]

            post_pred_student = model_post(post_pred_inp.detach())
            pred_angle_student = post_pred_student[:, : robot_config.angles_dim]

            pre_loss = criterion(pred_xyz, gt_xyz) + criterion(
                pred_rep, gt_rep
            )  # get_geodesic_loss(pred_rotmat, gt_rotmat)# criterion(pred_rep, gt_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
            post_loss = criterion(pred_angle_teacher, gt_angle) + criterion(pred_angle_student, gt_angle)

            optimizer_pre.zero_grad()
            pre_loss.backward()
            optimizer_pre.step()

            optimizer_post.zero_grad()
            post_loss.backward()
            optimizer_post.step()

            train_pre_loss += pre_loss.item() / len(train_dataloader)
            train_post_loss += post_loss.item() / len(train_dataloader)

        test_pre_loss = 0.0
        test_post_loss = 0.0
        model_pre.eval()
        model_post.eval()
        for sample in test_dataloader:
            with torch.no_grad():
                bs = sample["smpl_rep"].shape[0]
                pre_pred = model_pre(sample["smpl_rep"].float().to(device))
                pred_xyz = pre_pred[:, : robot_config.xyzs_dim]
                pred_rep = pre_pred[:, robot_config.xyzs_dim :]
                pred_rotmat = rotation_6d_to_matrix(pred_rep.reshape(bs, -1, 6))

                gt_rep = sample["robot_rep"].float().to(device)
                gt_rotmat = rotation_6d_to_matrix(gt_rep.reshape(bs, -1, 6))
                gt_xyz = sample["robot_xyz"].float().to(device)
                gt_angle = sample["robot_angle"].float().to(device)

                post_gt_inp = torch.cat([gt_xyz, gt_rep], dim=-1)
                post_pred_inp = torch.cat([pred_xyz, pred_rep], dim=-1)

                post_pred_teacher = model_post(post_gt_inp)
                pred_angle_teacher = post_pred_teacher[:, : robot_config.angles_dim]

                post_pred_student = model_post(post_pred_inp.detach())
                pred_angle_student = post_pred_student[:, : robot_config.angles_dim]

                pre_loss = criterion(pred_xyz, gt_xyz) + criterion(
                    pred_rep, gt_rep
                )  # get_geodesic_loss(pred_rotmat, gt_rotmat)# criterion(pred_rep, gt_rep) # get_geodesic_loss(pred_rotmat, gt_rotmat)
                post_loss = criterion(pred_angle_teacher, gt_angle) + criterion(pred_angle_student, gt_angle)

                test_pre_loss += pre_loss.item() / len(test_dataloader)
                test_post_loss += post_loss.item() / len(test_dataloader)

        print(
            "[EPOCH {}] tr loss : {:.03f},{:.03f} te loss :{:.03f},{:.03f}".format(
                epoch, train_pre_loss, train_post_loss, test_pre_loss, test_post_loss
            )
        )

        best_pre_loss = min(best_pre_loss, test_pre_loss)
        if best_pre_loss == test_pre_loss:
            torch.save(
                model_pre.state_dict(),
                f"out/models/{robot_config.robot_type.name}/human2{robot_config.robot_type.name}_best_pre_v1_filtered.pth",
            )

        best_post_loss = min(best_post_loss, test_post_loss)
        if best_post_loss == test_post_loss:
            torch.save(
                model_post.state_dict(),
                f"out/models/{robot_config.robot_type.name}/human2{robot_config.robot_type.name}_best_post_v1_filtered.pth",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_type",
        "-r",
        type=RobotType,
        default=RobotType.REACHY,
    )
    args: TrainArgs = parser.parse_args()

    train(args)
