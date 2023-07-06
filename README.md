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
  
  You can download data for rendering SMPL-X & VPoser Model via [this link](https://smpl-x.is.tue.mpg.de/download.php).
  
  You should move `bodymodel/` and `poser_v2_05/` into the `data/` folder.


## How to use codes.

1. run `src/process_data/sample.py` to sample the robot pose. 
```bash
python src/process_data/sample.py --max-seed 500 --num-per-seed 2000
```
- max_seed: The number of random seeds to be used for sampling
- num-per-seed: The number of poses to be sampled for each random seed.

2. run `src/process_data/fit2smpl.py` to fit human SMPL from the robot pose
```bash
python src/process_data/fit2smpl.py
```
- set --visualize 1 if you want to save the converted human SMPL as a data
  
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