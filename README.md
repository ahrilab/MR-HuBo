# retargeting4reachy

## Installation
- Prepare conda environment.
```
conda create rr38 -n python=3.8
conda activate rr38
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
git clone https://github.com/cotton-ahn/retargeting4reachy
cd retargeting4reachy
pip install -r requirements.txt
pip install ipython jupyter notebook
```
(+) install [PyRender](https://pyrender.readthedocs.io/en/latest/install/index.html#osmesa)
(+) prepare SMPL body model (.npz)
(+) download vposer_v2_05

## TODO LIST
- Try training retargeting network again.
    * sampling robot joint angle, position, and rotation representation.
    * SMPL joint info is axis angle. turn it into 6d representation.
    * improve the visualization part.
    * get familiar with MPII codes. 
