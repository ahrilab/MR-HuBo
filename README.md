# retargeting4reachy

## Installation
- Prepare conda environment.Tested on the recent PyTorch. Prepare the torch based on your GPU/CPU settings.
```bash
conda create rr38 -n python=3.8
conda activate rr38
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

- Clone the GitHub repository and install requirements.

```bash
git clone https://github.com/ahrilab/Retargeting4Reachy.git
# If you are using the fork option, please modify 'ahrilab' as 'your GitHub account username'.
cd retargeting4reachy
pip install -r requirements.txt
```

- Download data for SMPL
  
  You can download data for rendering SMPL-X & VPoser Model via [this link](https://smpl-x.is.tue.mpg.de/download.php).\
  You should move `bodymodel/` and `poser_v2_05/` into the `data/` folder.

- Download GT motions of robots

  You can download ground truth motions of robots via [this link](https://drive.google.com/file/d/102uf0paypd8zQCJhIqqBLtXoFDrjxh04/view?usp=sharing).\
  You should move this 'mr_gt.pkl' file into 'data/gt_motions/' path.

## Directory Structure
- data: Store the data of Robot's urdf, meshes, and motions data, VPoser & SMPL model, GT motions data.
- imgs: Images for README.md
- log: Log files for background python process (data generation, model training etc.). (git ignored)
- out: Outputs of code such as model weights, predicted motions, rendered videos, etc. (git ignored)
- src: codes for processing data, model defining, training, and inference, visualization, and utilities.
- tools: shell script for running code. (need to update)


## How to use codes.
> (We should reorganize this README file)

### (Temp block for Prof. Ahn) How to visualize the COMAN robot in Pybullet simulator.
- Run an interactive simulator
  ```bash
  python src/visualize/pybullet_interactive.py -r [robot_type] -v [view]
  ```

- Load a COMAN motion data and generate a video
  ```bash
  python src/visualize/pybullet_render.py -v VIEW --fps FPS [-s] -rp ROBOT_POSE_PATH -op OUTPUT_PATH
  ```

  Check each code file for details.


1. run `src/process_data/sample.py` to sample the robot pose. 
```bash
python src/process_data/sample.py -r REACHY
```

2. run `src/process_data/fit2smpl.py` to fit human SMPL from the robot pose
```bash
python src/process_data/fit2smpl.py -r REACHY
```
- add -viz arguments if you want to save the converted human SMPL as a data
  
3. run `src/process_data/adjust_neck.py` to make the neck angle the same as smpl
```bash
python src/process_data/adjust_neck.py
```

4. run `src/visualize/makevid_reachy.py` to visualize the robot's joint angle information.

5. run `src/visualize/makevid_pymaf.py` to visualize the result obtained from pymaf.


## Codes for model training (unorganized)
* src/model/train_human2reachy_dualmlp.py 
    - Use sampled ROBOT-HUMAN random pose information for training two MLP-based models
    - One model is called pre-model (model_pre). It converts human SMPL 6D REP into Reachy XYZ + 6D REP
    - Another model is called post-model (model_post). It converts Reachy XYZ + 6D REP to Reachy's angle.
    - It saves the model with the minimum test loss.

* src/model/test_human2reachy_dualmlp.py
    - It loads the pre- and post-models whose training procedure ended.
    - It reads the prediction results from pymaf as an input to pre- and post-models. 
    - And saves the angle result to ./output/pymaf_robot_v2.pkl.


## Work Flows
1. Try to sample out random robot pose.
2. Try to visualize one of the random robots poses in the video.
3. Try to convert the robot pose to human SMPL with V-Poser
4. Try to align neck information between reachy and human, by moving human neck info into reachy.
5. Try to train the model which converts XYZ+6D of humans to XYZ+6D of robots. 
6. Try to visualize the result. if you obtain `output/pymaf_robot_v2.pkl`, you can visualize with `src/visualize/makevid_reachy.py`

### Check "TODO" in the codes. I wrote some issues to be solved.