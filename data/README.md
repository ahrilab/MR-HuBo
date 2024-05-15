# Data Directory Structure
- bodymodel: Store the SMPL-X reutral.npz body model. You can download this via [this link](https://smpl-x.is.tue.mpg.de/download.php). (git ignored)
- vposer_v2_05: Model weight of VPoser. (git ignored)
- reachy, coman, nao: Robot's urdf, meshes, and motions data. Motion data is ignored in git system because its size is too big. You can generate the motion data with our code.
- gt_motions: Ground truth motion data. `mr_gt.pkl` is the robot motion data (joint angles $\mathbf{q}$), which is ignored in git system due to its size. You can download the robot motion file via [this link](https://drive.google.com/file/d/102uf0paypd8zQCJhIqqBLtXoFDrjxh04/view?usp=sharing). `amass_data` is SMPL parameters for the ground truth motions.
- temp: temporary folder (git ignored)
