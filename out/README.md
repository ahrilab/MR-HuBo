# Out Directory Structure

- gt_data: FK result and smpl fit result of Ground Truth (robot motion $\mathbf{q}$). `gt_smpl_vids` is a directory for rendered video of smpl fit parameters from each robot.
- models: Trained model weights for each robot.
    - _old: outdated model weight.
    - arm_only: model which use only SMPL arms joint representation as input.
        - ex: model applied extreme pose filtering.
        - no_ex: model not applied extreme pose filtering.
    - cf: model applied collision free. (full SMPL joints as input)
        - ex: model applied extreme pose filtering.
        - no_ex: model not applied extreme pose filtering.
    - no_cf: model not applied collision free. (full SMPL joints as input)
        - ex: model applied extreme pose filtering.
        - no_ex: model not applied extreme pose filtering.
- plot_spot: XYZSs positions and SMPL rendered result for each robot.
- pred_motions: Predicted motions ($\mathbf{q}$) and evaluated results. (same structure with the `models` directory)
- pybullet: pybullet rendered results in the same structure with the `models` directory.
