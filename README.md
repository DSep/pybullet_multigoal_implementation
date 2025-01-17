## DRL_Implementation

### Introduction
- This repository is a pytorch-based implementation of modern DRL algorithms.
- The package is under active development, use the latest version with caution
<!-- - Tested environments: Gym, Pybullet-gym, Pybullet-multigoal-gym -->
- *Priority is on continuous action algorithms*

#### Installation
Best usage is done via the parent repo: https://github.com/Paralelopipet/SEER. If you want to just try this code, you can, using:
```
git clone https://github.com/DSep/pybullet_multigoal_implementation.git
cd DRL_Implementation
python -m pip install -r requirements.txt
python -m pip install .
```
[Click here for example code](https://github.com/DSep/pybullet_multigoal_implementation/tree/master/drl_implementation/examples)
, to run the code you will need to install Gym, Pybullet, or pybullet-multigoal-gym. See env installation links below.
<!-- For more use cases, have a look at the [drl_imp_test repo](https://github.com/IanYangChina/drl_imp_test)\ -->
From the project root, run `python drl_implementation/examples/$SCRIPT_NAME.py`

##### State-based
- [X] DDPG - Deterministic, Continuous
- [X] TD3 -Deterministic, Continuous
- [X] SAC (Adaptive Temperature) - Stochastic, Continuous
<!-- - [ ] D4PG - Deterministic, Continuous ***-only works for torch 1.4.0, fix it later-*** -->
<!-- 
##### Image-based
- [ ] DQN - Deterministic, Discrete (LSTM network for Atari) ***-debugging-***
- [ ] SAC-AE - Continuous ***-to be built-***
- [ ] DrQ - Continuous ***-debugging-*** -->

##### Replay buffers
- [X] Hindsight
- [X] Prioritised

##### Tested Environments
<!-- - [X] [Pybullet Gym (Continuous)](https://github.com/bulletphysics/bullet3)
- [X] [OpenAI Gym Mujoco Robotics Multigoal Environment (Continuous)](https://openai.com/blog/ingredients-for-robotics-research/) -->
- [X] [Pybullet Multigoal Gym](https://github.com/IanYangChina/pybullet_multigoal_gym) (OpenAI Robotics 
Multigoal Pybullet Migration) (Continuous)

##### Some result figures
<img src="/src/figs.png" width="600"/>
<img src="/src/push.gif" width="600"/>
<img src="/src/pendulum.gif" width="600"/>

#### Reference Papers: Algorithm
* [DQN](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
* [DoubleDQN](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
* [LSTM network on raw Atari pixel observation](https://arxiv.org/pdf/1907.02908.pdf)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [TD3](https://arxiv.org/pdf/1802.09477.pdf)
* [SAC (Adaptive Temperature)](https://arxiv.org/pdf/1812.05905.pdf)
* [PER](https://arxiv.org/abs/1511.05952)
* [HER](http://papers.nips.cc/paper/7090-hindsight-experience-replay)
* [D4PG](https://arxiv.org/abs/1804.08617)
* [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX)
* [SAC-AE](https://arxiv.org/pdf/1910.01741.pdf)
* [DrQ](https://arxiv.org/abs/2004.13649)

#### Reference Papers: Implementation Matters
* [Time limit](https://arxiv.org/abs/1712.00378)
* [SOTA PPO Hyperparameters (many applicable to other algorithms)](https://arxiv.org/abs/2006.05990)
* [SAC Temperature Auto-tuning](https://arxiv.org/abs/1812.05905)
