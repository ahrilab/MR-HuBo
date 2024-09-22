# MR. HuBo: Redefining Data Pairing for Motion Retargeting Leveraging a Human Body Prior

Code repository for the paper:
**Redefining Data Pairing for Motion Retargeting Leveraging a Human Body Prior**
[Xiyana Figuera](https://github.com/xiyanafiguera), [Soogeun Park](https://github.com/bwmelon97), and [Hyemin Ahn](https://hyeminahn.oopy.io)

Our paper has been accepted @ IROS 2024! 🎉 \
Project Page: [🏠](https://sites.google.com/view/mr-hubo/home) | Arxiv Link: [📄]()

<!-- TODO: 페이퍼 Arxiv 링크, 웹사이트 링크 등 추가하기 -->

## Demo Videos
<p align="center">
<img src="./imgs/1_baseline.gif" height="400" />
</p>
<p align="center">
<img src="./imgs/4_Reachy.gif" height="400" />
</p>

## Installation & Setup
### Prepare conda environment and set pytorch and cuda environment.
```bash
conda create -n mr-hubo python=3.8
conda activate mr-hubo
conda install pytorch-cuda=11.8 cuda-toolkit=11.8 -c pytorch -c nvidia  # change the cuda version to match the version on your computer
```

### Clone the GitHub repository and install requirements.

```bash
git clone https://github.com/ahrilab/MR-HuBo.git
cd MR-HuBo
pip install -r requirements.txt
```

### Download SMPL-X models
  
  You can download SMPL-X & VPoser Model via [this link](https://smpl-x.is.tue.mpg.de/download.php).\
  We use the 'smplx neutral' model, and 'vposer_v2_05'.\
  Please make sure that you put the `bodymodel/smplx/neutral.npz` and `vposer_v2_05/` into the `data/` folder.

### Download GT motions of robots

  You can download ground truth motions of robots via [this link](https://drive.google.com/file/d/10V06RXFEpKfnt00iGE1qXnCQDbsODUqK/view?usp=sharing).\
  Please move this 'mr_gt.pkl' file into 'data/gt_motions/' path.

### Download AMASS dataset for GT motions of human

  You can download AMASS dataset via [this link](https://amass.is.tue.mpg.de/index.html).\
  Please download the 'CMU/SMPL-X N' data from the downloads tab.\
  Please move the motion files (e.g. `02_05_stageii.npz`) that we use for the ground truth into 'data/gt_motions/amass_data/'. You can see the motions used for GT in 'data/gt_motions/README.md'.

<!-- TODO: Add 'data/' directory structure -->

## Directory Structure
- data: Store the data of Robot's urdf, meshes, and motions data, VPoser & SMPL model, GT motions data.
- imgs: Images for README.md
- out: Outputs of code such as model weights, predicted motions, rendered videos, etc.
- src: Fragmented Codes, each individual file is responsible for a single function.
- tools: Integrated codes that perform each feature.


## How to use codes

### Generate \<Robot-Human\> Data for Training

```bash
python tools/generate_data.py -r [robot_type] -s [num_seeds] -p [poses_per_seed] -d [device] -i [restart_idx]

# example
python tools/generate_data.py -r COMAN
```


### Train the Motion Retargeting Network

```bash
python tools/train.py -r [robot_type] [-d <device>] [-n <num_data>] [-ef] [-os] [-w]

# example
python tools/train.py -r REACHY -ef -os -w
python tools/train.py -r COMAN -ef -d cuda:2
```

### Evaluation the Model

```bash
python tools/evaluate_model.py -r ROBOT_TYPE [-ef] [-os] [-d DEVICE] [-em EVALUATE_MODE]

# Example
python tools/evaluate_model.py -r REACHY
python tools/evaluate_model.py -r REACHY -ef -os -d cuda -em joint
```

### Visualize the Motion Retargeting Results

```bash
# Usage:
  python tools/render_robot_motion.py -r ROBOT_TYPE -mi MOTION_IDX [-ef] -e EXTENTION --fps FPS [-s]  # for pred_motion
  python tools/render_robot_motion.py -r ROBOT_TYPE -gt -mi MOTION_IDX -e EXTENTION --fps FPS [-s]    # for gt_motion

# Example:
  # render for prediction motion
  python tools/render_robot_motion.py -r COMAN -mi 13_08 -ef -e mp4 --fps 120 -s
  python tools/render_robot_motion.py -r COMAN -mi 13_18 -e mp4 --fps 120 -s

  # render for GT motion
  python tools/render_robot_motion.py -r=COMAN -gt -mi="13_08" -e mp4 --fps 120 -s
  python tools/render_robot_motion.py -r=COMAN -gt -mi="13_18" -e mp4 --fps 120
```

## Add New Robot Configuration
Mr. HuBo is general method which can be adapted to any humanoid robots, if a URDF (unified robot description format) of robot and scale factor for converting robot's position into SMPL position is given.

<!-- # TODO: 다른 로봇에 대한 데이터를 만들기 위한 방법 추가하기 -->



## Acknowledgements
Parts of the code are taken or adapted from the following repos:
<!-- TODO: Add items & link -->
- human-body-prior
- pymaf-x
- body-visualizer
<!-- Reachy 코드? -->

## Citations
```bibtext
@inproceedings{MR_HuBo:2024,
  title         = {Redefining Data Pairing for Motion Retargeting Leveraging Human Body Prior},
  author        = {Figuera, Xiyana and Park, Soogeun and Ahn, Hyemin.},
  year          = 2024,
  month         = october,
  booktitle     = {2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  event_place   = {Abu Zaby (Abu Dhabi), United Arab Emirates},
  month_numeric = 10
}
```

