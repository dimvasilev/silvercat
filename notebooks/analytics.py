# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This notebook performs the Analytics part of the task.
#
# Components:
# - Define what constitutes a "bad" outcome based on the dataset.
# - Explore how key variables impact the likelihood of bad outcomes.
# - Assess how historical credit risk policies have shaped the current portfolio composition.
# - Provide any additional insights that could influence credit risk decisioning.

# %%
# To explore
# - define outcome target
# - portfolio snapshot - distribution charts by key segment and over time
# - separation charts by key variables
# - impact of policy on portfolio composition

# %%
# import the required libraries
from pathlib import Path

import pandas as pd

from silvercat.utils import download_lending_club_data

# set options and global constants
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

PROJECT_ROOT = Path("/home/dimitar/Projects/silvercat")
DATA_DIR = PROJECT_ROOT / "data"
ACCEPTS = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
REJECTS = DATA_DIR / "rejected_2007_to_2018Q4.csv.gz"
DATA_DICT = PROJECT_ROOT / "LCDataDictionary.csv"

# %%
# import the datasets (if not available) using a little helper function
if not any(DATA_DIR.iterdir()):
    download_lending_club_data()
else:
    print("Data already downloaded")

# %%
# load a sample of accepts, rejects + the data dictionary files
accepts = pd.read_csv(ACCEPTS, nrows=100_000, low_memory=False)
rejects = pd.read_csv(REJECTS, nrows=100_000, low_memory=False)
data_dict = pd.read_csv(DATA_DICT)

# %%
accepts.shape

# %%
accepts.head(5)

# %%
accepts.describe()

# %%
accepts["loan_status"].value_counts()
# %%
# %%
# %%
# %%
# %%
