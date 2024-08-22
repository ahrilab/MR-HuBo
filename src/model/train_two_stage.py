import sys
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import os
from torch.utils.data import DataLoader
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm

sys.path.append("./src")
from model.net import MLP
from utils.RobotConfig import RobotConfig
from utils.consts import *


def train_two_stage(
    robot_config: RobotConfig,
    device: str,
    extreme_filter_off: bool,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    is_wandb: bool,
):
    """
    Train the two-stage model to predict robot joint angles from SMPL parameters.
    Save the best model weight for every 50 epochs.

    Args:
        robot_config (RobotConfig): Robot configuration
        device (str): Device for running the code
        extreme_filter (bool): Whether to use extreme filter or not
        train_dataloader (DataLoader): DataLoader for training data
        test_dataloader (DataLoader): DataLoader for testing data
        num_epochs (int): Number of epochs
        is_wandb (bool): Whether to use wandb or not
    """
    robot_name = robot_config.robot_type.name

    # hyperparameters
    dim_hidden = HIDDEN_DIM
    lr = LEARNING_RATE

    # create directory to save models
    model_save_dir = MODEL_WEIGHTS_DIR(robot_name, False, extreme_filter_off)
    os.makedirs(model_save_dir, exist_ok=True)

    # input & output dimensions
    # input: SMPL joint 6D representations (H)
    # output: robot joint angles (q)
    input_dim = SMPL_ARM_JOINT_REPS_DIM
    output_dim = robot_config.angles_dim

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
    criterion = nn.MSELoss()

    # train the model
    print("Start training...")
    for epoch in tqdm(range(num_epochs)):
        train_pre_loss = 0.0
        train_post_loss = 0.0
        model_pre.train()
        model_post.train()

        for sample in train_dataloader:
            if extreme_filter_off:
                # if extreme filter is not used, use the whole data
                smpl_rep = sample["smpl_rep"].float().to(device)
                gt_rep = sample["robot_rep"].float().to(device)
                gt_angle = sample["robot_angle"].float().to(device)

            else:
                # if extreme filter is used, sample the data with the probability of smpl_prob
                prob = sample["smpl_prob"]
                bernoulli_dist = Bernoulli(prob)

                # Sample from each Bernoulli distribution
                chosen_samples = bernoulli_dist.sample()

                # we use same batch size for each epoch whether we use extreme filter or not
                sample_index = (chosen_samples == 1).nonzero()[:EF_OFF_BATCH_SIZE, 0]

                ### We still need to do this for each sample[" "]
                # Extract the corresponding values in random_samples
                smpl_rep = sample["smpl_rep"][sample_index].float().to(device)
                gt_rep = sample["robot_rep"][sample_index].float().to(device)
                gt_angle = sample["robot_angle"][sample_index].float().to(device)

            # forward pass
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
        if is_wandb:
            # log the loss values to wandb
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

        # fmt: off
        pre_weight_name = MODEL_WEIGHT_NAME(robot_name, "pre", epoch // MODEL_SAVE_EPOCH)
        post_weight_name = MODEL_WEIGHT_NAME(robot_name, "post", epoch // MODEL_SAVE_EPOCH)
        pre_weight_path = os.path.join(model_save_dir, pre_weight_name)
        post_weight_path = os.path.join(model_save_dir, post_weight_name)
        # fmt: on

        best_pre_loss = min(best_pre_loss, test_pre_loss)
        if best_pre_loss == test_pre_loss:
            torch.save(model_pre.state_dict(), pre_weight_path)

        best_post_loss = min(best_post_loss, test_post_loss)
        if best_post_loss == test_post_loss:
            torch.save(model_post.state_dict(), post_weight_path)
