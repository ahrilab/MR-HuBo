# Model Code Directory Structure

- net: Defining the model.
- train_rep_only: main training code (generate trained model weights).
- test_rep_only: main inference code (generate prediected motions).
- pick_best_model: Find the best model weight using the validation GT motion set.
- evaluate_on_test_motions: Get the evaluation result from the test GT motion set using the best model weight.

---
- train_human2reachy_dualmlp.py: outdated traing code.
- test_human2reachy_dualmlp.py: outdated inference code.
