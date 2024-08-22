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


def train_one_stage(
    robot_config: RobotConfig,
    device: str,
    extreme_filter_off: bool,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    is_wandb: bool = False,
):
    """
    Train the one-stage model to predict robot joint angles from SMPL parameters.
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

    # input & output dimensions
    # input: SMPL joint 6D representations (H)
    # output: robot joint angles (q)
    input_dim = SMPL_ARM_JOINT_REPS_DIM
    output_dim = robot_config.angles_dim

    # create directory to save models
    model_save_dir = MODEL_WEIGHTS_DIR(robot_name, True, extreme_filter_off)
    os.makedirs(model_save_dir, exist_ok=True)

    # define model, optimizer, and loss function
    model = MLP(
        dim_input=input_dim,
        dim_output=output_dim,
        dim_hidden=dim_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    # train the model
    print("Start training...")
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        model.train()

        for sample in train_dataloader:
            if extreme_filter_off:
                # if extreme filter is not used, use the whole data
                smpl_rep = sample["smpl_rep"].float().to(device)
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
                gt_angle = sample["robot_angle"][sample_index].float().to(device)

            # forward pass
            pred_angle: torch.Tensor = model(smpl_rep)

            # fmt: off
            loss: torch.Tensor = criterion(pred_angle, gt_angle)

            # backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_dataloader)

        # Get test loss
        test_loss = 0.0
        model.eval()
        for sample in test_dataloader:
            with torch.no_grad():
                smpl_rep = sample["smpl_rep"].float().to(device)
                pred_angle = model(smpl_rep)
                gt_angle = sample["robot_angle"].float().to(device)

                loss = criterion(pred_angle, gt_angle)

                test_loss += loss.item() / len(test_dataloader)

        print(f"[EPOCH {epoch}] tr loss : {train_loss:.03f} te loss :{test_loss:.03f}")
        if is_wandb:
            # log the loss values to wandb
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )

        # Save the best model for every 50 epochs
        if epoch % MODEL_SAVE_EPOCH == 0:
            best_loss = 1e10

        weight_name = MODEL_WEIGHT_NAME(robot_name, "os", epoch // MODEL_SAVE_EPOCH)
        weight_path = os.path.join(model_save_dir, weight_name)

        best_loss = min(best_loss, test_loss)
        if best_loss == test_loss:
            torch.save(model.state_dict(), weight_path)
