#!/usr/bin/env python3

# %% libs

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import dataset_load_all, dataset_grid

# %% load
images, labels, mapping = dataset_load_all()

# %% DRAW GRID

dataset_grid(images, 0, 4).savefig("grid1.png", bbox_inches='tight', pad_inches=0)
dataset_grid(images, 4, 16).savefig("grid2.png", bbox_inches='tight', pad_inches=0)
dataset_grid(images, 16, 32).savefig("grid3.png", bbox_inches='tight', pad_inches=0)

# %%
