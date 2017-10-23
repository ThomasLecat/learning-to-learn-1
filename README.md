# Learning to Reinforcement Learn

Reproduction of the experiments 3.1.1 and 3.1.2 on meta deep reinforcement learning described in the paper "Learning to Reinforcement Learn" by Wang et. al

The A3C-LSTM agent is based on OpenAI's universe-starter-agent. It uses Tensorflow and OpenAI Gym libraries

The bandits environments used in the experiments can be found here:
https://github.com/ThomasLecat/gym-bandit-environments

# Experiments

Both experiments involve two armed bandits giving rewards of 1 with probabilities p1 and p2. In the first one, p1 and p2 are independent. In the second one, p1 and p2 are linked by the relation p1 + p2 = 1. The purpose of the experiments is to study whether the agent is able to learn the relationship between the two arms and use it to perform optimally. Please refer to the original paper for a full description of the experiments.

Bandits environments are stateless but the training is organised in fake episodes during which the internal state of the LSTM is kept. The length of these fake episode is 100 in these experiments but can be set to a different value using the argument -n (see "How to" section)

# running with bandit environments

Bandits environments are stateless, so there is now observation to input the agent's network. As described in the original paper Learning to Reinforcement Learn, the input of the agent's network when using bandit environments is a concatenation of :
* the last action
* the last reward
* the timestep within the (fake) episode

# How to

The two callable scripts are train.py and test.py.
Both take the same arguments as input. Among them, we can find:

* -w : number of workers working in parallel
* -e : gym environment id (ex: BanditTwoArmedDependentFixed-v0)
* -m : to resample environments at the beginning of each (fake) episode
* -n : number of trials in each (fake) episode (default is 100)
* -lr : learning rate (default is 1e-4)

Meta-learning is performed as soon as the -m argument is present. In that case, the environment is recreated at the beginning of each episode. As some parameters are stochastic and sampled at the creation of the environment, the MDP can change from one episode to another. This results in training (and / or testing) the agent on a set of MDPs instead of a single one.

Examples of calls:

1. Training and testing the agent on Pong (default env), no meta-learning, 16 workers: (similar behaviour to the original universe-starter-agent)

		python train.py -w 16

2. Training on a bandit env, no meta-learning, 1 worker, log directory changed:

		python train.py -e "BanditTwoArmedDependantUniform-v0" -l ./tmp/bandit

3. Training on a set of similar bandit env (meta-learning) and testing on a different bandit env, 16 worker, log-dir changed:

		python train.py -e "BanditTwoArmedDependantEasy-v0" -m -te "BanditTwoArmedDependantMedium-v0" -l ./tmp/bandit

# Hyperparameters

The hyperparameters are somewhat spread across the code... Here's the location of some of them :
* number of training steps : num_global_step in worker.py run function.
* number of testing steps : num_test_step next to num_global_step in worker_test.py
* discount factor : file A3C.py, class A3C, method process : change gamma value in the line : "batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)"
* number of steps in each rollout (t_max in the original A3C paper): file A3C.py, class A3c, method __init__, change the value in line : "num_local_step = 5"
* learning rate : change by adding the argument -lr <value> when calling python train.py (see section above)
* number of trials in a fake episode for bandit environments : change by adding the argument -n <value> when calling python trian.py (see section above)

# Dependencies

* Python 2.7 or 3.5
* [Golang](https://golang.org/doc/install)
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.12
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* libjpeg-turbo (`brew install libjpeg-turbo`)
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Installation

```
conda create --name learning-to-learn-1 python=3.5
source activate learning-to-learn-1

brew install tmux htop cmake golang libjpeg-turbo      # On Linux use sudo apt-get install -y tmux htop cmake golang libjpeg-dev

pip install "gym[atari]"
pip install universe
pip install six
pip install tensorflow
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy
```


Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate learning-to-learn-1```

# Example

`python train.py --num-workers 2 --env-id BanditTwoArmedDependantEasy-v0 --log-dir /tmp/banditDependent`

The code will launch the following processes:
* worker-0 - a process that runs policy gradient
* worker-1 - a process identical to process-1, that uses different random noise from the environment
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning
