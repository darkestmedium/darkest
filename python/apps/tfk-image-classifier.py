# Built-in imports
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")
import os
import random

# Third-party imports
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

from dataclasses import dataclass
import platform

# Darkest APi imports
import api.Darkest as da

# Text formatting
bold = "\033[1m"
end = "\033[0m"

block_plot=False



def set_seeds():
  # fix random seeds
  SEED_VALUE = 42

  random.seed(SEED_VALUE)
  np.random.seed(SEED_VALUE)
  tf.random.set_seed(SEED_VALUE)
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  
#     physical_devices = tf.config.list_physical_devices("GPU")
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         # Invalid device or cannot modify virtual devices once initialized.
#         pass

  return

set_seeds()

# Creating a MirroredStrategy for distributed training.
# This strategy effectively replicates the model's layers on each GPU or other available devices,
# syncing their weights after each training step.
DISTRIBUTE_STRATEGY = tf.distribute.MirroredStrategy()

# Printing the number of devices that are in sync with the MirroredStrategy.
# This indicates how many replicas of the model are being trained in parallel.
print('Number of devices: {}'.format(DISTRIBUTE_STRATEGY.num_replicas_in_sync))


# If required, update the root_dir path according to the dataset path.

root_dir = "/home/darkest/Dropbox/code/dataset/classification/opencv-TF-course-project-1-image-classification/dataset"
# root_dir = r"../input/opencv-TF-course-project-1-image-classification/dataset"

train_dir = os.path.join(root_dir, "Train")
valid_dir = os.path.join(root_dir, "Valid")


def list_folders(startpath):
  for root, _, files in os.walk(startpath):
    level = root.replace(startpath, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f'{indent}{os.path.basename(root):<8}')


list_folders(root_dir)

print(f"{bold}Training Classes:{end} ")
for i in os.listdir(train_dir):
  print(i)
    
print("------------")

print(f"{bold}Validation Classes:{end} ")
for j in os.listdir(valid_dir):
  print(j)



num_train_files = 0
num_valid_files = 0

### YOUR CODE HERE

###

print(f"{bold}Number of Training samples: {end}{num_train_files}")
print(f"{bold}Number of Validation samples: {end}{num_valid_files}")