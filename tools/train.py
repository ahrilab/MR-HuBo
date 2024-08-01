"""
Train the model to predict robot joint angles from SMPL parameters.

Usage:
    python tools/train.py -r [robot_type] [-d <device>] [-n <num_data>] [-ef-off] [-os] [-w]

Example:
    python tools/train.py -r REACHY -os -w
    python tools/train.py -r COMAN -d cuda:2
"""

import argparse
import wandb
import time
import sys
from torch.utils.data import DataLoader

sys.path.append("./src")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TrainArgs
from utils.consts import *
from utils.data import load_and_split_train_test, H2RMotionData
from model.train_one_stage import train_one_stage
from model.train_two_stage import train_two_stage


def train(args: TrainArgs):
    robot_config = RobotConfig(args.robot_type)

    # wandb init
    if args.wandb:
        wandb.init(project="mr_hubo")
        current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        if args.one_stage:
            if args.extreme_filter_off:
                run_name = f"human2{args.robot_type.name}_os_noef_{current_time}"
            else:
                run_name = f"human2{args.robot_type.name}_os_ef_{current_time}"
        else:
            if args.extreme_filter_off:
                run_name = f"human2{args.robot_type.name}_ts_noef_{current_time}"
            else:
                run_name = f"human2{args.robot_type.name}_ts_ef_{current_time}"
        wandb.run.name = run_name

    # set hyperparameters
    if args.extreme_filter_off:
        num_epochs = EF_OFF_NUM_EPOCHS
        batch_size = EF_OFF_BATCH_SIZE
    else:
        num_epochs = EF_EPOCHS
        batch_size = EF_BATCH_SIZE

    # load data
    num_data = args.num_data
    # fmt: off
    input_path  = robot_config.SMPL_PARAMS_PATH  # input: SMPL parameters        (H)
    reps_path   = robot_config.XYZS_REPS_PATH    # target: robot xyzs and reps   (R)
    target_path = robot_config.ANGLES_PATH       # target: robot joint angles    (q)
    # fmt: on

    robot_xyzs, robot_reps, robot_angles, smpl_reps, smpl_prob = (
        load_and_split_train_test(
            input_path=input_path,
            reps_path=reps_path,
            target_path=target_path,
            num_data=num_data,
            split_ratio=DATA_SPLIT_RATIO,
            extreme_filter_off=args.extreme_filter_off,
        )
    )

    train_dataset = H2RMotionData(
        robot_xyzs["train"],
        robot_reps["train"],
        robot_angles["train"],
        smpl_reps["train"],
        smpl_prob["train"],
        args.extreme_filter_off,
    )
    test_dataset = H2RMotionData(
        robot_xyzs["test"],
        robot_reps["test"],
        robot_angles["test"],
        smpl_reps["test"],
        smpl_prob["test"],
        args.extreme_filter_off,
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # train model
    if args.one_stage:
        train_one_stage(
            robot_config=robot_config,
            device=args.device,
            extreme_filter_off=args.extreme_filter_off,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_epochs=num_epochs,
            is_wandb=args.wandb,
        )
    else:
        train_two_stage(
            robot_config=robot_config,
            device=args.device,
            extreme_filter_off=args.extreme_filter_off,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_epochs=num_epochs,
            is_wandb=args.wandb,
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
        "--device",
        "-d",
        type=str,
        default=DEVICE,
        help="Device to run the model",
    )
    parser.add_argument(
        "--extreme-filter-off",
        "-ef-off",
        action="store_true",
        help="train model with extreme filter",
    )
    parser.add_argument(
        "--one-stage",
        "-os",
        action="store_true",
        help="whether train one stage model",
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="Use wandb to log training process",
    )
    parser.add_argument(
        "--num-data",
        "-n",
        type=int,
        default=NUM_SEEDS,
        help="Number of data to train",
    )

    args: TrainArgs = parser.parse_args()
    train(args)
