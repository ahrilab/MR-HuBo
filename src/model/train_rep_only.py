"""
Train the model to predict robot joint angles from SMPL parameters.

Usage:
    python src/model/train_rep_only.py -r [robot_type] [-w]

Example:
    python src/model/train_rep_only.py -r REACHY -w
"""

import argparse
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("./src")
from utils.data import split_train_test, H2RMotionData
from model.net import MLP
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TrainArgs
from utils.consts import *


def train(args: TrainArgs):
    # hyperparameters
    num_data = NUM_SEEDS
    split_ratio = 50

    dim_hidden = HIDDEN_DIM
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    device = DEVICE

    if args.wandb:
        wandb.init(project="mr_hubo")
        current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        run_name = f"human2{args.robot_type.name}_rep_only_{current_time}"
        wandb.run.name = run_name

    # prepare dataset
    robot_config = RobotConfig(args.robot_type)

    # create directory to save models
    model_save_path = f"out/models/{robot_config.robot_type.name}"
    os.makedirs(model_save_path, exist_ok=True)

    # fmt: off
    input_path = robot_config.ROBOT_TO_SMPL_PATH    # input: SMPL parameters
    target_path = robot_config.TARGET_DATA_PATH     # target: robot joint angles
    # fmt: on

    robot_xyzs, robot_reps, robot_angles, smpl_reps, smpl_rots = split_train_test(
        input_path, target_path, num_data, split_ratio, False
    )

    train_dataset = H2RMotionData(
        robot_xyzs["train"],
        robot_reps["train"],
        robot_angles["train"],
        smpl_reps["train"],
        smpl_rots["train"],
    )
    test_dataset = H2RMotionData(
        robot_xyzs["test"],
        robot_reps["test"],
        robot_angles["test"],
        smpl_reps["test"],
        smpl_rots["test"],
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # define model, optimizer, and loss function
    model_pre = MLP(
        dim_input=SMPL_JOINT_REPS_DIM,
        dim_output=robot_config.reps_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=robot_config.angles_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    optimizer_pre = optim.Adam(model_pre.parameters(), lr, weight_decay=1e-6)
    optimizer_post = optim.Adam(model_post.parameters(), lr, weight_decay=1e-6)

    # train the model
    best_pre_loss = 1e10
    best_post_loss = 1e10
    criterion = nn.MSELoss()

    print("Start training...")
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_pre_loss = 0.0
        train_post_loss = 0.0
        model_pre.train()
        model_post.train()
        for sample in train_dataloader:
            # forward pass
            smpl_rep = sample["smpl_rep"].float().to(device)
            pred_rep = model_pre(smpl_rep)

            gt_rep = sample["robot_rep"].float().to(device)
            gt_angle = sample["robot_angle"].float().to(device)

            teacher_angle = model_post(gt_rep)
            student_angle = model_post(pred_rep.detach())

            # fmt: off
            pre_loss = criterion(pred_rep, gt_rep)
            post_loss = (
                criterion(teacher_angle, gt_angle) + 
                criterion(student_angle, gt_angle)
            )
            # fmt: on

            # backprop and update parameters
            optimizer_pre.zero_grad()
            pre_loss.backward()
            optimizer_pre.step()

            optimizer_post.zero_grad()
            post_loss.backward()
            optimizer_post.step()

            train_pre_loss += pre_loss.item() / len(train_dataloader)
            train_post_loss += post_loss.item() / len(train_dataloader)

        # Get test loss
        test_pre_loss = 0.0
        test_post_loss = 0.0
        model_pre.eval()
        model_post.eval()
        for sample in test_dataloader:
            with torch.no_grad():
                smpl_rep = sample["smpl_rep"].float().to(device)
                pred_rep = model_pre(smpl_rep)

                gt_rep = sample["robot_rep"].float().to(device)
                gt_angle = sample["robot_angle"].float().to(device)

                teacher_angle = model_post(gt_rep)
                student_angle = model_post(pred_rep.detach())

                # fmt: off
                pre_loss = criterion(pred_rep, gt_rep)
                post_loss = (
                    criterion(teacher_angle, gt_angle) + 
                    criterion(student_angle, gt_angle)
                )
                # fmt: on

                test_pre_loss += pre_loss.item() / len(test_dataloader)
                test_post_loss += post_loss.item() / len(test_dataloader)

        print(
            "[EPOCH {}] tr loss : {:.03f},{:.03f} te loss :{:.03f},{:.03f}".format(
                epoch, train_pre_loss, train_post_loss, test_pre_loss, test_post_loss
            )
        )
        if args.wandb:
            wandb.log(
                {
                    "train_pre_loss": train_pre_loss,
                    "train_post_loss": train_post_loss,
                    "test_pre_loss": test_pre_loss,
                    "test_post_loss": test_post_loss,
                }
            )

        # Save the best model for every 50 epochs
        if epoch % 50 == 0:
            best_pre_loss = 1e10
            best_post_loss = 1e10

        best_pre_loss = min(best_pre_loss, test_pre_loss)
        if best_pre_loss == test_pre_loss:
            torch.save(
                model_pre.state_dict(),
                f"{model_save_path}/human2{robot_config.robot_type.name}_rep_only_pre_{epoch//50}.pth",
            )

        best_post_loss = min(best_post_loss, test_post_loss)
        if best_post_loss == test_post_loss:
            torch.save(
                model_post.state_dict(),
                f"{model_save_path}/human2{robot_config.robot_type.name}_rep_only_post_{epoch//50}.pth",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_type",
        "-r",
        type=RobotType,
        default=RobotType.REACHY,
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="Use wandb to log training process",
    )
    args: TrainArgs = parser.parse_args()

    train(args)
