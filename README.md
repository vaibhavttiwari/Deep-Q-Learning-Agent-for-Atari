# Deep-Q-Learning-Agent-for-Atari
  This repository contains a Deep Q Learning Agent that can be trained on Atari games. This is an implementation of Deepmind' [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf).
  
# Dependencies
+ tensorflow
+ numpy
+ gym
+ skimage
+ matplotlib

# Testing
Run the following command in the terminal : 

`python main.py`

This will train the agent of the default environment (SpaceInvaders-v0) for 50 episodes.

The file *config.json* contains the configuration of the model which can be changed according to the user.

Run `python main.py --help` for more information about the parameters.
