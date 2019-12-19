# Related To Training

LOAD_CHECKPOINTS = False
USE_CUDA = True
N_CRITIC = 5 # Train critic this many times more than generator
GP_WEIGHT = 10
INITIAL_PROGRESS = 1 # How many layers to use when training starts

# Training

LOAD_CHECKPOINTS = False
ITERATIONS = 2000
BATCH_SIZE = 16
SAMPLE_INTERVAL = 50
CHECKPOINT_INTERVAL = 100
GROW_INTERVAL = 50

# Related To General Model Stuff

DECONV_MODE = "upsample"
USE_DROPOUT = True
NORM = "sigmoid"
CHANNELS = 3
IMG_SIZE = 128
LEARNING_RATE = 1e-3
