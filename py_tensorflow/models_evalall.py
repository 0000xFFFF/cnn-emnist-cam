#!/usr/bin/env python3

# %% libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import dataset_load_all

import gc
from tensorflow.keras.models import load_model
from utils_tf import prepdata

images, labels, mapping = dataset_load_all()
test_input, test_target = prepdata(images, labels)
del images
gc.collect()
del labels
gc.collect()

models_dir = "models"
for i, file in enumerate(os.listdir(models_dir)):
    print(f"model: {file} -- ", end='')
    full_path = os.path.join(models_dir, file)
    model = load_model(full_path)
    model.evaluate(test_input, test_target, verbose=2)
