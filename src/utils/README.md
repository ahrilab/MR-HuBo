# Utility Code Directory Structure

- consts: Constants for the whole code. (divide them into each robot, smpl and common constants)
- data: Codes for loading the data files and construct a Dataset class instance.
- evaluate: Return the evaluation result when it inputs the pred_motion and gt_motion.
- forward_kinematics: Return the Forward Kinematics results when it inputs the kinematics chain and angles list.
- hbp: Codes for VPoser IK Engine and SMPL rendering.
- RobotConfig: Robot Configuration Class which assign the constants for each robot.
- transform: Codes for transformming rotation matrix, quaternion, and 6D representation.
- types: Type definition for Enum classes and Arguments.

---
### Outdated
- loss: geodesic_loss defined
- viz: rendering code for kinpy
