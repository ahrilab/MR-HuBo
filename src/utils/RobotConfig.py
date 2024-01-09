import sys

sys.path.append("./src")
from utils.types import RobotType
from utils.consts import *


class RobotConfig:
    def __init__(self, robot_type: RobotType) -> None:
        # Check the robot type is a member of the robot types
        assert (
            robot_type in RobotType._member_map_.values()
        ), f"robot type should be a member of ({RobotType._member_names_})."

        self.robot_type = robot_type

        # Set the config as the reachy robot
        if robot_type == RobotType.REACHY:
            self.URDF_PATH = REACHY_URDF_PATH
            self.RAW_DATA_PATH = REACHY_RAW_PATH
            self.FIX_DATA_PATH = REACHY_FIX_PATH
            self.ROBOT_TO_SMPL_PATH = REACHY_SMPL_PATH
            self.joi_range = REACHY_JOI_RANGE
            self.joi_keys = REACHY_JOI_KEYS
            self.smpl_joint_idx = REACHY_SMPL_JOINT_IDX
            self.convert_xyzs = reachy_xyzs_to_smpl_xyzs
            self.xyzs_dim = REACHY_XYZS_DIM
            self.reps_dim = REACHY_REPS_DIM
            self.angles_dim = REACHY_ANGLES_DIM
            self.smpl_reps_dim = REACHY_SMPL_REPS_DIM

        elif robot_type == RobotType.COMAN:
            self.URDF_PATH = COMAN_URDF_PATH
            self.RAW_DATA_PATH = COMAN_RAW_PATH
            self.FIX_DATA_PATH = COMAN_FIX_PATH
            self.ROBOT_TO_SMPL_PATH = COMAN_SMPL_PATH
            self.joi_range = COMAN_JOI_RANGE
            self.joi_keys = COMAN_JOI_KEYS
            self.smpl_joint_idx = COMAN_SMPL_JOINT_IDX
            self.convert_xyzs = coman_xyzs_to_smpl_xyzs
            self.xyzs_dim = COMAN_XYZS_DIM
            self.reps_dim = COMAN_REPS_DIM
            self.angles_dim = COMAN_ANGLES_DIM
            self.smpl_reps_dim = COMAN_SMPL_REPS_DIM

        # Please add here if you want to add a new robot type
        # elif robot_type == RobotType.NEW_ROBOT:
        #     ...

        # If the robot type is not a member of the robot types
        else:
            print("Warning: It must be a problem !!")
