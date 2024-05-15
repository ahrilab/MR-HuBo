# Model Code Directory Structure

- net: Defining the model.
- train_two_stage: main training code for two-staged network (generate trained model weights for pre and post network).
- train_one_stage: training code for one-staged network.
- infer_with_two_stage: main inference code using two-staged network (generate prediected motions).
- infer_with_one_stage: inference code using one-staged network.
- pick_best_model: Find the best model weight using the validation GT motion set.
- evaluate_on_test_motions: Get the evaluation result from the test GT motion set using the best model weight.
