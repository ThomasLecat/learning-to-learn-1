class Config:

    #########################################################################
    # Hyperparameters:

    # Number of steps
    NUM_TRAINING_STEPS = 2000000
    NUM_TEST_STEPS = 30000

    # Learning-rate for the Adam optimizer
    LEARNING_RATE=1e-4

    # Discount factor
    DISCOUNT = 0.99

    # Tmax
    '''Number of steps before the network is updated (on each thread)'''
    TIME_MAX = 5

    # Coefficient of the value fonction loss
    BETA_V = 0.05 # 0.5

    # Coefficient of the entropy regularization loss
    BETA_E_START = 1.0 # 0.2
    BETA_E_END = 0.0

    # Input of the neural network for non-bandit environmets (ex: Atari games)
    IMAGE_WIDTH = 42
    IMAGE_HEIGHT = 42
    ''' 42x42 is OK for Pong. For other games, use 84x84'''

    # Choose optimizer [ADAM, RMSPROP]
    OPTIMIZER = 'ADAM'

    # Reward clipping
    USE_REWARD_CLIPPING = False
    REWARD_MIN = -1
    REWARD_MAX = 1
    ''' Do not clip reward for experiments on bandit environments from paper:
    "Learning to Reinforcement Learn" '''

    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_NORM = 40.0
