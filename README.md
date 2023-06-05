# retargeting4reachy

## Installation
- Prepare conda environment.Tested on the recent pytorch. Prepare the torch based on your GPU/CPU settings.
```
conda create rr38 -n python=3.8
conda activate rr38
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
git clone https://github.com/cotton-ahn/retargeting4reachy
cd retargeting4reachy
pip install -r requirements.txt
pip install ipython jupyter notebook
```

- install [PyRender](https://pyrender.readthedocs.io/en/latest/install/index.html#osmesa)

## How to use codes.
1. run `reachy/sample.py` to sample robot pose. 
```
python reachy/sample.py --max-seed 500 --num-per-seed 2000
```
- max_seed : how many random seed to be used for sampling
- num-per-seed: how many poses to be sampled for each random seed.

2. run `reachy/fit2smpl.py` to fit human SMPL from robot pose
```
python reachy/fit2smpl.py
```
- set --visualize 1 if you want to save the converted human SMPL as a data
    
3. run `reachy/adjust_neck.py` to make neck angle same as smpl
```
python reachy/adjust_neck.py
```

4. run `reachy/makevid.py` to visualize the robot's joint angle information.

5. run `human/makevid_pymaf.py` to visualize the result obtained from pymaf.


## Codes for model training (unorganized)
* train_human2reachy_dualmlp.py 
    - Use sampled ROBOT-HUMAN random pose information for training two MLP-based models
    - One model is called as pre- model. (model_pre). It converts human SMPL 6D REP into Reachy XYZ + 6D REP
    - Another model is called as post-model (model_post). It converts Reachy XYZ + 6D REP to Reachy's angle.
    - It saves the model with the minimum test loss.

* test_human2reachy_dualmlp.py
    - It loads the pre- and post- models whose training procedure ended.
    - It reads the prediction result from pymaf as an input to pre- and post- models. 
    - And saves the angle result to ./pymaf_robot_v2.pkl     


## Things I do not remember well...
1. human/makevid_param.py
    * It would be for visualizing SMPL parameters into video, which is not necessarily to have same format as the result from PyMAF-X

2. playground*.py
    * Several things I tried for fun (?) 


## Work Flows
1. Try to sample out random robot pose.
2. Try to visualize one of the random robot pose into video.
3. Try to convert the robot pose to human SMPL with V-Poser
4. Try to align neck information between reachy and human, by moving human neck info into reachy.
5. Try to train the model which converts XYZ+6D of human to XYZ+6D of robot. 
6. Try to visualize the result. if you obtain `pymaf_robot_v2.pkl`, you can visualize with `reachy/makevid.py`

### Check "TODO" in the codes. I wrote some issues to be solved.