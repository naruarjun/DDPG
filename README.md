# DDPG
DDPG implementaion with Hindsight Experience Replay (HER) on bare Tensorflow

## Setting up:
1. Clone the repository
```bash
git clone https://github.com/abhayraw1/DDPG.git
```
2. Create a virtual environment. 
It's optional but *highly* recommended to do so. 
You can create virtual environments using `virtualenv` or `conda`. 
Make sure you create the environment for `python3`.

3. Install the necessary dependencies in your virtual environment. Given below is a list of them.
  ```yaml
  tensorflow # for neural nets and stuff
  tensorboard # for visualization purposes
  pyyaml # for parsing yaml config files
  gym # open-ai gym
  ```
## References
1. Deep Deterministic Policy Gradients Paper: [Continuous control with deep reinforcement learning
](https://arxiv.org/abs/1509.02971)
2. Hindsight Experience Replay Paper: [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
