#!/usr/bin/env python3

# %% libs
import gc
from utils_tf import prepdata
from utils_tf_selectmodel import selectmodel

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import load_XXp


# %% select model

model = selectmodel()

# %% new model settings
print("loading...")
images, labels = load_XXp("20")

# %% preprocess set for tf
print("preparing images for tensorflow...")
test_input, test_target = prepdata(images, labels)
del images
gc.collect()
del labels
gc.collect()
print("images prepared.")

# %% eval model
print("EVALUATE: ")
results = model.evaluate(test_input, test_target, verbose=2)

# %%
