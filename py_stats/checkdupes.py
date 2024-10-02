#!/usr/bin/env python3

# %% libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_load import dupes_check_all

# %% check dupes all

dupes_check_all()

# %%
