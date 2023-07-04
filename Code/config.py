CHANNEL_N = 4        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40 
BATCH_SIZE = 8 
POOL_SIZE = 4
CHEMISTRY_RATE=0.1 
ENERGY_RATE=0.1 
CELL_FIRE_RATE = 1.0
TARGET_MODE = "Changing_Corners" #"Random Energy", "Corners","Curriculum Barriers", "Barriers",

import numpy as np
RNG = np.random.default_rng(seed=27)
