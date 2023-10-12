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

        # Set the config as the reachy robot
        if robot_type == RobotType.REACHY:
            self.URDF_PATH = REACHY_URDF_PATH
            self.RAW_DATA_PATH = REACHY_RAW_PATH
            self.FIX_DATA_PATH = REACHY_FIX_PATH
            self.joint_range = REACHY_JOINT_RANGE
            self.convert_xyzs = convert_reachy_xyzs_to_smpl_xyzs

        elif robot_type == RobotType.COMAN:
            pass

        # Please add here if you want to add a new robot type
        # elif robot_type == RobotType.NEW_ROBOT:
        #     self.URDF_PATH = NEW_ROBOT_URDF_PATH
        #     self.RAW_DATA_PATH = NEW_ROBOT_RAW_PATH
        #     self.FIX_DATA_PATH = NEW_ROBOT_FIX_PATH
        #     self.joint_range = NEW_ROBOT_JOINT_RANGE
        #     self.convert_xyzs = convert_new_robot_xyzs_to_smpl_xyzs

        # If the robot type is not a member of the robot types
        else:
            print("Warning: It must be a problem !!")
