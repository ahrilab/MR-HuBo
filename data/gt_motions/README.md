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
- Pepper
- Atlas
- Nao

## Motions
```
- 02_05: punch strike
- 13_15: laugh
- 13_18: boxing
- 14_05: unscrew bottlecap, drink soda      # not exists in AMASS
- 14_10: wash windows                       # not exists in AMASS
- 14_24: direct traffic, wave, point        # not exists in AMASS
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
[-4.67364407e-02 -1.40284715e-02  1.28603314e-01  3.94823323e-01 1.05771326e+00  1.45845627e-01 -8.17219453e-01  1.55053216e-01 -1.10746113e+00 -5.93442511e-02 -5.22422310e-01 -1.35674356e-17 -6.95733594e-18]
```
