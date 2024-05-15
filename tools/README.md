# Tools: Integrated codes (executable codes) that perform each feature.

- generate_data.py: Generate <Robot-Human> paired pose data
- train.py: Train the model to predict robot joint angles from SMPL parameters.
- evaluate_model.py: Picks the best model on the validation set and evaluates it on the test motions.
- render_robot_motion.py: Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.