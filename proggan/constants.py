# Related To Training

LOAD_CHECKPOINTS = False
USE_CUDA = True
N_CRITIC = 5 # Train critic this many times more than generator
GP_WEIGHT = 10
BATCH_SIZE = 128
SAMPLE_INTERVAL = 100
CHECKPOINT_INTERVAL = 250
EPOCHS = 2000
INITIAL_PROGRESS = 1 # How many layers to use when training starts
PROG_STEP = 200 # When to up "progression"
ALPHA = 0.5 # Weight of new model

# Related To General Model Stuff

DECONV_MODE = "upsample"
USE_DROPOUT = True
NORM = "sigmoid"
CHANNELS = 3
IMG_SIZE = 64

# Specific Model Conv stuff

MAX_FILTERS = 256 # Highest filter count used in Gen/Disc
MIN_FILTERS = 64 # Lowest filter count used in Gen/Disc 
