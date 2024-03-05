"""
Train the model to predict robot joint angles from SMPL parameters.

Usage:
    python src/model/train_rep_only.py -r [robot_type] [-w] [-cf] [-ef]

Example:
    python src/model/train_rep_only.py -r REACHY -w -cf -ef
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
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm

sys.path.append("./src")
from utils.data import load_and_split_train_test, H2RMotionData
from model.net import MLP
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TrainArgs
from utils.consts import *


def train(args: TrainArgs):
    robot_config = RobotConfig(args.robot_type)

    # hyperparameters
    num_data = NUM_SEEDS
    split_ratio = 50

    dim_hidden = HIDDEN_DIM
    lr = LEARNING_RATE
    device = DEVICE

    if args.extreme_filter:
        num_epochs = EF_EPOCHS
        batch_size = EF_BATCH_SIZE
    else:
        num_epochs = NUM_EPOCHS
        batch_size = BATCH_SIZE

    # input & output dimensions
    # input: SMPL joint 6D representations (H)
    # output: robot joint angles (q)
    if args.arm_only:
        input_dim = SMPL_ARM_JOINT_REPS_DIM
    else:
        input_dim = SMPL_JOINT_REPS_DIM

    if args.collision_free:
        output_dim = robot_config.cf_angles_dim
    else:
        output_dim = robot_config.angles_dim

    if args.wandb:
        wandb.init(project="mr_hubo")
        current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        if args.collision_free:
            if args.extreme_filter:
                run_name = f"human2{args.robot_type.name}_cf_ef_{current_time}"
            else:
                run_name = f"human2{args.robot_type.name}_cf_noef_{current_time}"
        else:
            if args.extreme_filter:
                run_name = f"human2{args.robot_type.name}_nocf_ef_{current_time}"
            else:
                run_name = f"human2{args.robot_type.name}_nocf_noef_{current_time}"
        wandb.run.name = run_name

    # create directory to save models
    if args.collision_free:
        if args.extreme_filter:
            model_save_path = f"out/models/{robot_config.robot_type.name}/cf/ex/"
        else:
            model_save_path = f"out/models/{robot_config.robot_type.name}/cf/no_ex/"
    else:
        if args.extreme_filter:
            model_save_path = f"out/models/{robot_config.robot_type.name}/no_cf/ex/"
        else:
            model_save_path = f"out/models/{robot_config.robot_type.name}/no_cf/no_ex/"

    # fmt: off
    if args.arm_only:
        if args.extreme_filter:
            model_save_path = f"out/models/{robot_config.robot_type.name}/arm_only/ex/"
        else:
            model_save_path = f"out/models/{robot_config.robot_type.name}/arm_only/no_ex/"
    # fmt: on

    os.makedirs(model_save_path, exist_ok=True)

    # load data
    if args.collision_free:
        SMPL_PARAMS_PATH = robot_config.CF_SMPL_PARAMS_PATH
        XYZS_REPS_PATH = robot_config.CF_XYZS_REPS_PATH
        ANGLES_PATH = robot_config.CF_ANGLES_PATH
    else:
        SMPL_PARAMS_PATH = robot_config.SMPL_PARAMS_PATH
        XYZS_REPS_PATH = robot_config.XYZS_REPS_PATH
        ANGLES_PATH = robot_config.ANGLES_PATH

    # fmt: off
    input_path  = SMPL_PARAMS_PATH  # input: SMPL parameters        (H)
    reps_path   = XYZS_REPS_PATH    # target: robot xyzs and reps   (R)
    target_path = ANGLES_PATH       # target: robot joint angles    (q)
    # fmt: on

    robot_xyzs, robot_reps, robot_angles, smpl_reps, smpl_prob = (
        load_and_split_train_test(
            input_path=input_path,
            reps_path=reps_path,
            target_path=target_path,
            num_data=num_data,
            split_ratio=split_ratio,
            collision_free=args.collision_free,
            extreme_filter=args.extreme_filter,
            arm_only=args.arm_only,
        )
    )

    train_dataset = H2RMotionData(
        robot_xyzs["train"],
        robot_reps["train"],
        robot_angles["train"],
        smpl_reps["train"],
        smpl_prob["train"],
        args.extreme_filter,
    )
    test_dataset = H2RMotionData(
        robot_xyzs["test"],
        robot_reps["test"],
        robot_angles["test"],
        smpl_reps["test"],
        smpl_prob["test"],
        args.extreme_filter,
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # define model, optimizer, and loss function
    model_pre = MLP(
        dim_input=input_dim,
        dim_output=robot_config.reps_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=output_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    optimizer_pre = optim.Adam(model_pre.parameters(), lr, weight_decay=1e-6)
    optimizer_post = optim.Adam(model_post.parameters(), lr, weight_decay=1e-6)

    best_pre_loss = 1e10
    best_post_loss = 1e10
    criterion = nn.MSELoss()

    # train the model
    print("Start training...")
    for epoch in tqdm(range(num_epochs)):
        train_pre_loss = 0.0
        train_post_loss = 0.0
        model_pre.train()
        model_post.train()

        for sample in train_dataloader:
            # forward pass
            if args.extreme_filter:
                prob = sample["smpl_prob"]
                bernoulli_dist = Bernoulli(prob)

                # Sample from each Bernoulli distribution
                chosen_samples = bernoulli_dist.sample()

                # we use same batch size for each epoch whether we use extreme filter or not
                sample_index = (chosen_samples == 1).nonzero()[:BATCH_SIZE, 0]

                ### We still need to do this for each sample[" "]
                # Extract the corresponding values in random_samples
                smpl_rep = sample["smpl_rep"][sample_index].float().to(device)
                gt_rep = sample["robot_rep"][sample_index].float().to(device)
                gt_angle = sample["robot_angle"][sample_index].float().to(device)

            else:
                smpl_rep = sample["smpl_rep"].float().to(device)
                gt_rep = sample["robot_rep"].float().to(device)
                gt_angle = sample["robot_angle"].float().to(device)

            pred_rep: torch.Tensor = model_pre(smpl_rep)
            teacher_angle = model_post(gt_rep)
            student_angle = model_post(pred_rep.detach())

            # fmt: off
            pre_loss: torch.Tensor = criterion(pred_rep, gt_rep)
            post_loss: torch.Tensor = (
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
        if epoch % MODEL_SAVE_EPOCH == 0:
            best_pre_loss = 1e10
            best_post_loss = 1e10

        if args.arm_only:
            pre_weight_name = f"human2{robot_config.robot_type.name}_arm_only_pre_{epoch // MODEL_SAVE_EPOCH}.pth"
            post_weight_name = f"human2{robot_config.robot_type.name}_arm_only_post_{epoch // MODEL_SAVE_EPOCH}.pth"
        else:
            pre_weight_name = f"human2{robot_config.robot_type.name}_pre_{epoch // MODEL_SAVE_EPOCH}.pth"
            post_weight_name = f"human2{robot_config.robot_type.name}_post_{epoch // MODEL_SAVE_EPOCH}.pth"

        pre_weight_path = os.path.join(model_save_path, pre_weight_name)
        post_weight_path = os.path.join(model_save_path, post_weight_name)

        best_pre_loss = min(best_pre_loss, test_pre_loss)
        if best_pre_loss == test_pre_loss:
            torch.save(model_pre.state_dict(), pre_weight_path)

        best_post_loss = min(best_post_loss, test_post_loss)
        if best_post_loss == test_post_loss:
            torch.save(model_post.state_dict(), post_weight_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_type",
        "-r",
        type=RobotType,
        default=RobotType.REACHY,
    )
    parser.add_argument(
        "--collision_free",
        "-cf",
        action="store_true",
        help="use collision-free data",
    )
    parser.add_argument(
        "--extreme-filter",
        "-ef",
        action="store_true",
        help="train model with extreme filter",
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="Use wandb to log training process",
    )
    parser.add_argument(
        "--arm_only",
        "-a",
        action="store_true",
        help="train model with SMPL's arm joint only",
    )
    args: TrainArgs = parser.parse_args()

    train(args)
