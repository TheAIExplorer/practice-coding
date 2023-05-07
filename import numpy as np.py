import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from importlib import reload

# enable automatic reloading of modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# enable interactive shell
InteractiveShell.ast_node_interactivity = 'all'


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
