#!/usr/bin/env python3

# %% libs
from utils_tf_selectmodel import selectmodel
import time
import numpy as np

import matplotlib.pyplot as plt
from utils_tf import prepdata

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import dataset_loadset, dataset_img
from emnist_maps import emnist_class_mapping_reversed


# %% select model

model = selectmodel("all_v3_batch10000_epoch5.keras")

# %% load data for testing

test_images, test_labels, test_mapping = dataset_loadset("digits", "test")
test_input, test_target = prepdata(test_images, test_labels)

# %% eval model

print("EVALUATE: ")
results = model.evaluate(test_input, test_target, verbose=2)

# %% testing model

print("predicting...")
predicted_labels = np.argmax(model.predict(test_input, verbose=2), axis=1)
print("done.")

# %% find misses

a_lst = np.array([emnist_class_mapping_reversed[i] for i in predicted_labels])

right = 0
wrong = 0
for i, (a, b) in enumerate(zip(a_lst, test_labels)):
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        wrong += 1
    else:
        right += 1

l = test_labels.shape[0]
print(f"right predictions: {right}/{l} -- {right/l}")
print(f"wrong predictions: {wrong}/{l} -- {wrong/l}")

# %% see some misses

for i, (a, b) in enumerate(zip(a_lst, test_labels)):
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(test_images, test_labels, i, note=f"predict: {a}")
        time.sleep(2)

# %% single image by index

index = 400
image = test_input[index:index+1]
print(image.shape)
val = np.argmax(model.predict(image), axis=1)
value = val[0]

label = emnist_class_mapping_reversed[value]
print("predict:", label)

plt.figure()
plt.imshow(image[0])
plt.show()

# %% 
