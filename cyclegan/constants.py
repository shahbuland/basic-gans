# Dataset info

IMG_SIZE = 64
CHANNELS = 3

# Hyperparams

LEARNING_RATE = 2e-4
BETAS = (0.5,0.9)

# Training

CYCLE_WEIGHT = 0.5
ITERATIONS = 5000
BATCH_SIZE = 16
USE_CUDA = True
SAMPLE_INTERVAL = 25
CHECKPOINT_INTERVAL = 200
N_CRITIC = 1
LOAD_CHECKPOINTS = False
