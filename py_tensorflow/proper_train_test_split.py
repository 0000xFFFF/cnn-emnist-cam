#!/usr/bin/env python3

# %% libs

import math
import random as rand
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import dataset_load_all, save_XXp


def charimgs_empty():
    charimgs = {chr(i): [] for i in range(48, 58)}        # '0'-'9'
    charimgs.update({chr(i): [] for i in range(97, 123)}) # 'a'-'z'
    charimgs.update({chr(i): [] for i in range(65, 91)})  # 'A'-'Z'
    return charimgs


def get_charimgs():
    images, labels, mapping = dataset_load_all()
    n = labels.shape[0]
    charimgs = charimgs_empty()

    for i in range(n):
        charimgs[labels[i]].append(images[i])

    return charimgs


def save_to_numpy(charimgs, prefix):
    images = []
    labels = []

    for k, v in charimgs.items():
        images.extend(v)
        labels.extend([k] * len(v))

    images = np.array(images)
    labels = np.array(labels)

    save_XXp(images, labels, prefix)


# %% sample

rand_seed = 47
print(f"using rand seed: {rand_seed}")
rand.seed(rand_seed)

charimgs = get_charimgs()
charimgs_80p = charimgs_empty()
charimgs_20p = charimgs_empty()

for k, v in charimgs.items():
    charimgs_len = len(v)
    if charimgs_len > 0:  # Ensure there are elements to sample from
        indices = list(range(charimgs_len))
        rand.shuffle(indices)
        p = math.floor(0.8 * charimgs_len)
        sampled_indices_80p = indices[:p]
        sampled_indices_20p = indices[p:]
        charimgs_80p[k].extend([v[i] for i in sampled_indices_80p])
        charimgs_20p[k].extend([v[i] for i in sampled_indices_20p])
        print(f"{k} : {charimgs_len} --> 80% {p}, 20% {len(sampled_indices_20p)}")
    else:
        print(f"{k} : {charimgs_len} --> no samples available")


save_to_numpy(charimgs_80p, '80p')
save_to_numpy(charimgs_20p, '20p')

# %%
