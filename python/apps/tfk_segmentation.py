# Built-in imports
import os
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")  # import api 'cause hate venvs
import random
import argparse
import platform
import math

# Third-party imports

import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import zipfile 
import requests
import albumentations as A
import os

# from tensorflow.keras.utils import Sequence
from dataclasses import dataclass



# Darkest APi imports
from api import np
from api import cv
from api import tf
from api import tfk
from api import qtc

import api.Darkest as da
import api.DarkestMl as daml



DISTRIBUTE_STRATEGY = tf.distribute.MirroredStrategy()



@dataclass(frozen=True)
class DatasetConfig:
  NUM_CLASSES: int = 10
  IMG_WIDTH:   int = 256
  IMG_HEIGHT:  int = 256

  # DATA_TRAIN_IMAGES: str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/train/images/*.jpg'
  # DATA_TRAIN_LABELS: str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/train/masks/*.bmp'
  # DATA_VALID_IMAGES: str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/valid/images/*.jpg'
  # DATA_VALID_LABELS: str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/valid/masks/*.bmp'
  # DATA_TEST_IMAGES:  str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/test/images/*.jpg'
  # DATA_TEST_LABELS:  str = '/home/darkest/Dropbox/code/resources/dataset/segmentation/dataset_SUIM/test/masks/*.bmp'


  DATA_TRAIN_IMAGES: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/dataset/train/images/*.jpg"
  DATA_TRAIN_LABELS: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/dataset/train/masks/*.png"
  DATA_VALID_IMAGES: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/dataset/train/images/*.jpg"
  DATA_VALID_LABELS: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/dataset/train/masks/*.png"
  DATA_TEST_IMAGES:  str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/dataset/train/images/*.jpg"
  # DATA_TEST_LABELS:  str = f'{root_dir}/test/masks/*.png'






class CustomSegmentationDataset(tfk.utils.Sequence):

  def __init__(self, *, batch_size, image_size, image_paths, mask_paths, num_classes, apply_aug):
    """Generic Dataset class for semantic segmentation datasets.

    Arguments:
      batch_size:  Number of samples to be included in each batch of data.
      image_size:  Image and mask size to be used for training.
      image_paths: Path to image directory.
      mask_paths:  Path to masks directory.
      num_classes: Total number of classes present in dataset.
      apply_aug:   Should augmentations be applied.

    """
    self.batch_size  = batch_size
    self.image_size  = image_size
    self.image_paths = image_paths
    self.mask_paths  = mask_paths
    self.num_classes = num_classes
    self.aug = apply_aug
    
    self.x = np.empty((self.batch_size,) + self.image_size + (3,), dtype="float32")
    self.y = np.empty((self.batch_size,) + self.image_size, dtype="float32")
    
    if self.aug:
      self.train_transforms = self.transforms()
    
    self.resize_transforms = self.resize()


  def __len__(self):
    return self.mask_paths.__len__() // self.batch_size


  def transforms(self):
    # Data augmentation.
    train_transforms = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0.2, shift_limit=0.2, p=0.5, border_mode=0)
    ])
    return train_transforms


  def resize(self):
    resize_transforms = A.Resize(
      height=self.image_size[0], width=self.image_size[1],
      interpolation=cv.INTER_NEAREST,
      always_apply=True, p=1
    )
    return resize_transforms


  def reset_array(self):
    self.x.fill(0.)
    self.y.fill(0.)


  def __getitem__(self, idx):
    self.reset_array()
    i = idx * self.batch_size
    batch_image_paths = self.image_paths[i : i + self.batch_size]
    batch_mask_paths = self.mask_paths[i : i + self.batch_size]
    
    for j, (input_image, input_mask) in enumerate(zip(batch_image_paths, batch_mask_paths)):
      # Read the image and convert to RGB.
      img = cv.imread(input_image)
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   
      # Read the mask and convert to RGB.
      msk = cv.imread(input_mask)
      msk = cv.cvtColor(msk, cv.COLOR_BGR2RGB)

      # Resize the image and mask.
      resized  = self.resize_transforms(image=img, mask=msk)
      img, msk = resized['image'], resized['mask']
      
      if self.aug:
        # Apply augmentations.
        train_augment = self.train_transforms(image=img, mask=msk)
        img, msk = train_augment['image'], train_augment['mask']

      # Store image in x.
      self.x[j] = img / 255. # Normalizing image to be in range [0.0, 1.0]
      
      # Convert RGB segmentation mask to multi-channel (one-hot) encoded arrays where 
      # each channel represents a single class whose pixel values are either 0 or 1, 
      # where a 1 represents a pixel location associated with the class that corresponds 
      # to the channel.
      msk = rgb_to_onehot(msk)
      
      # Convert the multi-channel (one-hot encoded) mask to a single channel (grayscale)   
      # representation whose values contain the class IDs for each class (essentially 
      # collapsing the one-hot encoded arrays into a single channel).
      self.y[j] = msk.argmax(-1)
      
    return self.x, self.y



# id2color = {
#   0: (0,  0,    0),    # BW: Background/waterbody 
#   1: (0,  0,    255),  # HD: Human divers 
#   2: (0,  255,  255),  # WR: Wrecks and ruins
#   3: (255, 0,   0),    # RO: Robots and instruments
#   4: (255, 0,   255),  # RI: Reefs and invertebartes
#   5: (255, 255, 0),    # FV: Fish and vertebrates
#  }

id2color = {
  0: (0, 0, 0),  # Background
  1: (255, 0, 0), # Building Flooded
  2: (200, 90, 90), # Non-Flooded Building
  3: (128, 128, 0), # Road Flooded 
  4: (155, 155, 155),  # Non-Flooded Road
  5: (0, 255, 255),  # Water
  6: (55, 0, 255),  # Tree
  7: (255, 0, 255),  # Vehicle
  8: (245, 245, 0),  # Pool
  9: (0, 255, 0),  # Grass
 }



# Function to one-hot encode RGB mask labels.
def rgb_to_onehot(rgb_arr, color_map=id2color, num_classes=DatasetConfig.NUM_CLASSES):
  shape = rgb_arr.shape[:2] + (num_classes,)
  arr = np.zeros( shape, dtype=np.float32 )

  for i, classes in enumerate(color_map):
    arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])

  return arr



# Function to convert a single channel mask representation to an RGB mask.
def num_to_rgb(num_arr, color_map=id2color):
  single_layer = np.squeeze(num_arr)
  output = np.zeros(num_arr.shape[:2]+(3,))
  
  for k in color_map.keys():
    output[single_layer==k] = color_map[k]
      
  return np.float32(output) / 255. # return a floating point array in range [0.0, 1.0]



# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
  alpha = 1.0 # Transparency for the original image.
  beta  = 0.7 # Transparency for the segmentation map.
  gamma = 0.0 # Scalar added to each sum.
  
  segmented_image = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
  
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  
  image = cv.addWeighted(image, alpha, segmented_image, beta, gamma, image)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  
  return np.clip(image, 0.0, 1.0)





def display_image_and_mask(data_list, color_mask=False, color_map=id2color):
  plt.figure(figsize=(16, 6))
  title = ['GT Image', 'GT Mask', 'Overlayed Mask']

  grayscale_gt_mask = data_list[1]
  
  # Create RGB segmentation map from grayscale segmentation map.
  rgb_gt_mask = num_to_rgb(data_list[1], color_map=color_map)
  
  # Create the overlayed image.
  overlayed_image = image_overlay(data_list[0], rgb_gt_mask)
  
  data_list.append(overlayed_image)
  
  for i in range(len(data_list)):
    plt.subplot(1, len(data_list), i+1)
    plt.title(title[i])
    if title[i] == 'GT Mask':
      if color_mask:
        plt.imshow(np.array(rgb_gt_mask))
      else:
        plt.imshow(np.array(grayscale_gt_mask))
    else:
      plt.imshow(np.array(data_list[i]))
    plt.axis('off')

  plt.show()




def create_datasets(aug=False, split_ratio:float=0.8):
  # Training image and mask paths.
  data_images = sorted(glob.glob(f"{DatasetConfig.DATA_TRAIN_IMAGES}"))
  data_masks  = sorted(glob.glob(f"{DatasetConfig.DATA_TRAIN_LABELS}"))

  split_index = int(data_images.__len__() * split_ratio)

  train_images = data_images[:split_index]
  train_masks = data_masks[:split_index]

  valid_images = data_images[split_index:]
  valid_masks  = data_masks[split_index:]

  # Train data loader.
  train_ds = CustomSegmentationDataset(
    batch_size=8,
    image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
    image_paths=train_images,
    mask_paths=train_masks,
    num_classes=DatasetConfig.NUM_CLASSES,
    apply_aug=aug,
  )

  # Validation data loader.
  valid_ds = CustomSegmentationDataset(
    batch_size=8,
    image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
    image_paths=valid_images,
    mask_paths=valid_masks,
    num_classes=DatasetConfig.NUM_CLASSES,
    apply_aug=False,
  )

  return train_ds, valid_ds









def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/models/", help="Path to the model file and checkpoint.")
  parser.add_argument("-dfp", "--datasetFilePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/Project_3_FloodNet_Dataset/", help="Path to the dataset.")
  parser.add_argument("-ts", "--targetSize", type=tuple, default=(256, 256), help="Target size.")
  parser.add_argument("-ep", "--epochs", type=int, default=1, help="Number of epochs to train for.")
  parser.add_argument("-lr", "--learningRate", type=float, default=0.01, help="Learning rate.")
  parser.add_argument("-bs", "--batchSize", type=int, default=4*DISTRIBUTE_STRATEGY.num_replicas_in_sync, help="Batch size.")
  parser.add_argument("-mp", "--multiProcessing", type=bool, default=True if platform.system()=="Linux" else False, help="Wheter or not to use multi threading.")
  parser.add_argument("-now", "--numberOfWorkers", type=int, default=4, help="Number of workers to use for training.")
  parser.add_argument("-noc", "--numberOfClasses", type=int, default=10, help="Number of classes to use for training / classification.")
  parser.add_argument("-da", "--dataAugmentation", type=bool, default=True, help="Wheter or not to use data augmentation.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()

  train_ds, valid_ds = create_datasets(aug=True)

  for i, (images, masks) in enumerate(valid_ds):
    if i == 3: break
    # Retrieve last image in data batch as an example.
    image, mask = images[-1], masks[-1]
    display_image_and_mask([image, mask], color_mask=True)

