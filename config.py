class Config:

    #########################################################################
    # Hyperparameters:

    # Number of steps
    NUM_TRAINING_STEPS = 50000000
    NUM_TEST_STEPS = 1000000

    # Learning-rate for the Adam optimizer
    LEARNING_RATE=1e-4

    # Discount factor
    DISCOUNT = 0.99

    # Tmax
    '''Number of steps before the network is updated (on each thread)'''
    TIME_MAX = 5

    # Entropy regularization coefficient
    BETA_START = 0.02
    BETA_END = 0.0

    # Input of the neural network
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
