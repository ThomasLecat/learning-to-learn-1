# Learning to Reinforcement Learn

Reproduction of the experiments 3.1.1 and 3.1.2 on meta deep reinforcement learning described in the paper "Learning to Reinforcement Learn" by Wang et. al

The A3C-LSTM agent is based on OpenAI's universe-starter-agent. It uses Tensorflow and OpenAI Gym libraries

The bandits environments used in the experiments can be found here:
https://github.com/ThomasLecat/gym-bandit-environments

# Description of the experiments

[See original paper for full description]
Both experiments involve two armed bandits. The two arms give a reward of 1 with probabilities p1 and p2.

In the first experiment, p1 and p2 are independent: p1 ~ [0,1] and p2 ~ [0,1].

In the second experiment, p1 and p2 are linked by the relation p2 = 1 - p1.
Different configurations are possibles : p1 ~ U[0,1] (uniform setup), p1 ~ {0.1,0.9} (easy setup), p1 ~ {0.25,0.75} (medium setup), p1 ~ {0.4,0.6} (hard setup).

The purpose of the experiments is to study whether the agent is able to learn the relationship between the two arms and use it to perform optimally.

Bandits environments are stateless but the training is organised in fake episodes during which the internal state of the LSTM is kept. The length of these fake episode is 100 in these experiments but can be set to a different value using the argument -n when calling train.py or test.py

### Note : distinctive features of bandit environments

Bandits environments are stateless, so there is now observation to input the agent's network. As described in the original paper Learning to Reinforcement Learn, the input of the agent's network when using bandit environments is a concatenation of :
* the last action
* the last reward
* the timestep within the (fake) episode

# How to

First, set the values of the parameters you want in config.py

Then call train.py to train the model and test.py to test it.
Both scripts take the same arguments as input. Among them, we can find:

* -w : number of workers running in parallel
* -e : gym environment id (ex: BanditTwoArmedDependentFixed-v0 or Breakout-v0)
* -m : to recreate the environment at the beginning of each episode (performs meta learning if the creation of the environment bears some randomness)
* -n : number of trials in each episode for bandits environments (default is 100)

Meta-learning is performed as soon as the -m argument is present. In that case, the environment is recreated at the beginning of each episode. As some parameters are stochastic and sampled at the creation of the environment, the MDP can change from one episode to another. This results in training (and / or testing) the agent on a set of MDPs instead of a single one.

# Examples:

1. Training the agent with 1 worker on a two armed bandit with independent arms:

`python train.py -e "BanditTwoArmedIndependentUniform-v0" -l ./tmp/banditsIndependent -m`

2. Training the agent with 2 workers on a two armed bandit with dependent arms and p1 ~ {0.1,0.9}

`python train.py -w 2 -e "BanditTwoArmedDependantUniform-v0" -l ./tmp/banditEasy -m`

3. Testing the previous agent on a two armed bandit with dependent arms and p2 ~ {0.25,0.75}

`python test.py -e "BandittwoArmedDependentMedium-v0" -l ./tmp/banditEasy -m`

4. Training an agent on Pong (default env, no meta learning), 16 workers:

`python train.py -w 16 -e "Pong-v0" -l ./tmp/pong`

The code will launch the eighteen following processes:
* worker-0 - a process that runs policy gradient
			⋮
* worker-15 - identical to process-1, uses different random noise from the environment
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning

# Results

### Experiment 3.1.1

### Experiment 3.1.2

As in the original paper, the agent can generalize (to some extent) from one setup to another.
For exemple: An agent trained on the easy setup performs reasonably well on the medium setup.

Trained and tested on the easy setup | Tested on the medium setup
:-----------------------------------:|:-----------------------------------:
![](https://i.imgur.com/mwcflcM.png) | ![](https://i.imgur.com/AS9zXFI.png)

# Dependencies & installation

### Dependencies

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

### Installation

```
conda create --name learning-to-learn python=3.5
source activate learning-to-learn

# On Mac:
brew install tmux htop cmake golang libjpeg-turbo      
# On Linux:
sudo apt-get install -y tmux htop cmake golang libjpeg-dev zlib1g-dev

pip install "gym[atari]"
pip install universe
pip install six
pip install tensorflow
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy
```


Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate learning-to-learn```

# Example

`python train.py --num-workers 2 --env-id BanditTwoArmedDependantEasy-v0 --log-dir /tmp/banditDependent`
