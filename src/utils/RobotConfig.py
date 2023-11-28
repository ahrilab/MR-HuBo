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

        elif robot_type == RobotType.COMAN:
            self.URDF_PATH = COMAN_URDF_PATH
            self.RAW_DATA_PATH = COMAN_RAW_PATH
            self.FIX_DATA_PATH = COMAN_FIX_PATH
            self.ROBOT_TO_SMPL_PATH = COMAN_SMPL_PATH
            self.joi_range = COMAN_JOI_RANGE
            self.joi_keys = COMAN_JOI_KEYS
            self.smpl_joint_idx = COMAN_SMPL_JOINT_IDX
            self.convert_xyzs = coman_xyzs_to_smpl_xyzs

        elif robot_type == RobotType.YUMI:
            self.URDF_PATH = YUMI_URDF_PATH
            self.joint_keys = []

        # Please add here if you want to add a new robot type
        # elif robot_type == RobotType.NEW_ROBOT:
        #     self.URDF_PATH = NEW_ROBOT_URDF_PATH
        #     self.RAW_DATA_PATH = NEW_ROBOT_RAW_PATH
        #     self.FIX_DATA_PATH = NEW_ROBOT_FIX_PATH
        #     self.joint_range = NEW_ROBOT_JOINT_RANGE
        #     self.joint_keys = NEW_ROBOT_JOINT_RANGE.keys()
        #     self.smpl_joint_idx = NEW_ROBOT_SMPL_JOINT_IDX
        #     self.convert_xyzs = convert_new_robot_xyzs_to_smpl_xyzs

        # If the robot type is not a member of the robot types
        else:
            print("Warning: It must be a problem !!")
