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
            self.ANGLES_PATH = REACHY_ANGLES_PATH
            self.XYZS_REPS_PATH = REACHY_XYZS_REPS_PATH
            self.SMPL_PARAMS_PATH = REACHY_SMPL_PARAMS_PATH

            self.joi_range = REACHY_JOI_RANGE
            self.joi_keys = REACHY_JOI_KEYS
            self.cf_joi_keys = REACHY_CF_JOI_KEYS
            self.smpl_joint_idx = REACHY_SMPL_JOINT_IDX
            self.convert_xyzs = reachy_xyzs_to_smpl_xyzs
            self.xyzs_dim = REACHY_XYZS_DIM
            self.reps_dim = REACHY_REPS_DIM
            self.angles_dim = REACHY_ANGLES_DIM
            self.smpl_reps_dim = REACHY_SMPL_REPS_DIM
            self.cf_angles_dim = REACHY_CF_ANGLES_DIM
            self.evaluate_links = REACHY_EVALUATE_LINKS
            self.joint_vectors = REACHY_JOINT_VECTORS

        elif robot_type == RobotType.COMAN:
            self.URDF_PATH = COMAN_URDF_PATH
            self.URDF_4_RENDER_PATH = COMAN_URDF_4_RENDER_PATH
            self.ANGLES_PATH = COMAN_ANGLES_PATH
            self.XYZS_REPS_PATH = COMAN_XYZS_REPS_PATH
            self.SMPL_PARAMS_PATH = COMAN_SMPL_PARAMS_PATH

            self.joi_range = COMAN_JOI_RANGE
            self.joi_keys = COMAN_JOI_KEYS
            self.cf_joi_keys = COMAN_CF_JOI_KEYS
            self.smpl_joint_idx = COMAN_SMPL_JOINT_IDX
            self.convert_xyzs = coman_xyzs_to_smpl_xyzs
            self.xyzs_dim = COMAN_XYZS_DIM
            self.reps_dim = COMAN_REPS_DIM
            self.angles_dim = COMAN_ANGLES_DIM
            self.smpl_reps_dim = COMAN_SMPL_REPS_DIM
            self.cf_angles_dim = COMAN_CF_ANGLES_DIM
            self.evaluate_links = COMAN_EVALUATE_LINKS
            self.joint_vectors = COMAN_JOINT_VECTORS

        elif robot_type == RobotType.NAO:
            self.URDF_PATH = NAO_URDF_PATH
            self.ANGLES_PATH = NAO_ANGLES_PATH
            self.XYZS_REPS_PATH = NAO_XYZS_REPS_PATH
            self.SMPL_PARAMS_PATH = NAO_SMPL_PARAMS_PATH

            self.joi_range = NAO_JOI_RANGE
            self.joi_keys = NAO_JOI_KEYS
            self.cf_joi_keys = NAO_CF_JOI_KEYS
            self.smpl_joint_idx = NAO_SMPL_JOINT_IDX
            self.convert_xyzs = nao_xyzs_to_smpl_xyzs
            self.exclude_links = NAO_EXCLUDE_LINKS
            self.xyzs_dim = NAO_XYZS_DIM
            self.reps_dim = NAO_REPS_DIM
            self.angles_dim = NAO_ANGLES_DIM
            self.smpl_reps_dim = NAO_SMPL_REPS_DIM
            self.cf_angles_dim = NAO_CF_ANGLES_DIM
            self.evaluate_links = NAO_EVALUATE_LINKS
            self.joint_vectors = NAO_JOINT_VECTORS

        # Please add here if you want to add a new robot type
        # elif robot_type == RobotType.NEW_ROBOT:
        #     ...

        # If the robot type is not a member of the robot types
        else:
            print("Warning: It must be a problem !!")
