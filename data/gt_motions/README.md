## Download GT file link
> 추가하기

## load pickle file
```python
import pickle
motion_data = pickle.load(open("data/gt_motions/mr_gt.pkl", "rb"))
```

## How is the dictionary structured?
> `motion_data[robot_name][motion_id][original_or_cf][number_of_frame]`

## Robots
- Reachy
- Coman
- Nao

## Motions
```
- 02_05: punch strike
- 13_08: unscrew bottlecap, drink soda, screw on bottlecap
- 13_15: laugh
- 13_18: boxing
- 13_21: wash windows
- 13_28: direct traffic, wave, point
- 15_08: hand signals - horizontally revolve forearms
- 26_02: basketball signals
- 54_16: superhero
- 55_16: panda (human subject)
- 56_02: vignettes - fists up, wipe window, yawn, stretch, angrily grab, smash against wall
```

## Original or cf
- Original: `q`
- Collision free: `q_cf`

## Example
```python
motion_data["Coman"]["56_02"]["q_cf"][0]
```

Output:
```
<수정하기>
[-4.67364407e-02 -1.40284715e-02  1.28603314e-01  3.94823323e-01 1.05771326e+00  1.45845627e-01 -8.17219453e-01  1.55053216e-01 -1.10746113e+00 -5.93442511e-02 -5.22422310e-01 -1.35674356e-17 -6.95733594e-18]
```
