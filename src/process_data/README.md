# Process Data Code Directory Structure

- sample: sample random robot joint angles ($\mathbf{q}$) in valid range, get the FK result ($P, R$) of the angles, and save the angles & fk_results.
- fit2smpl: Get the SMPL parameter ($H$) using VPoser from converted position of the robot ($P$).
- adjust_nect: Adjust the neck joint of robot. (only for Reachy)
- convert_mat2pkl: Convert `.mat` file into `.pkl`, which is the format of our angles data.
- fk_with_angles: Get FK results from angles (`.pkl`) file.
