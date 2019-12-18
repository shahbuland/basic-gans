# Related To Training

LOAD_CHECKPOINTS = False
USE_CUDA = False
N_CRITIC = 5 # Train critic this many times more than generator
GP_WEIGHT = 10
INITIAL_PROGRESS = 1 # How many layers to use when training starts
PROG_STEP = 200 # When to up "progression"
ALPHA = 0.5 # Weight of new model

# Training

LOAD_CHECKPOINTS = False
ITERATIONS = 2000
BATCH_SIZE = 8
SAMPLE_INTERVAL = 1
CHECKPOINT_INTERVAL = 1
GROW_INTERVAL = 50

# Related To General Model Stuff

DECONV_MODE = "upsample"
USE_DROPOUT = True
NORM = "sigmoid"
CHANNELS = 3
IMG_SIZE = 64
LEARNING_RATE = 1e-4
