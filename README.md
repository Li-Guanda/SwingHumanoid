# SwingHumanoid
 Self-Organizing Neural Network for Reproducing Human Postural Mode Alternation through Deep Reinforcement Learning 

1. Install IsaacGym from:
https://developer.nvidia.com/isaac-gym/download
2: Install IsaacGymEnvs from:
https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
3: Merge the files in this project into IsaacGymEnvs
4: pip install matplotlib, pandas
5: run python train.py task=Swing2 headless=False in the terminal for training
6: run python train.py task=Swing2test test=True checkpoint=[path] in the terminal for testing the results
