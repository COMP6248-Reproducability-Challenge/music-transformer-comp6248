# Filename: test_JSB.py
# Author: zckoh
# Date Created: 03-Mar-2019 3:47:28 pm
# Description: Simple script to test on how to load in the dataset



import numpy as np

# Load in the JSB chorales dataset.
data = np.load('Jsb16thSeparated.npz', fix_imports=True, encoding="latin1")

train_data = data['train']
validation_data = data['valid']
test_data = data['test']

# Now the dataset is in numpy arrays for us to use.
print(type(train_data))
print(train_data[0].shape)
