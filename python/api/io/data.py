# Built-in imports
import logging
import pickle
import subprocess
import platform
from typing import overload, Union, Sequence

# Third-party imports
from api import qtc
from api import tfk
from api import cv
from api import np
from api import alb




class DatasetIO():
  """Import wrapper for the api.io.data.DataIO class.
  """


  @classmethod
  def augmentation(cls):
    """Combines multiple augmentations in a single processing pipeline.

    Reference:
      https://www.tensorflow.org/tutorials/images/data_augmentation#data_augmentation_2

    """
    return tfk.Sequential([
      tfk.layers.RandomFlip("horizontal"),
      tfk.layers.RandomFlip("vertical"),
      tfk.layers.RandomRotation(0.2),
      tfk.layers.RandomZoom(0.2),
      tfk.layers.RandomContrast(0.2),
      tfk.layers.RandomBrightness(0.2),
      tfk.layers.Rescaling(1./255)
    ])









class SegmentationDataset(tfk.utils.Sequence):

  def __init__(self, *, batch_size, image_size, image_paths, mask_paths, num_classes, color_map, apply_aug):
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
    self.color_map = color_map
    self.aug = apply_aug
    
    self.x = np.empty((self.batch_size,) + self.image_size + (3,), dtype="float32")
    self.y = np.empty((self.batch_size,) + self.image_size, dtype="float32")
    
    if self.aug:
      self.train_transforms = self.transforms()
    
    self.resize_transforms = self.resize()


  def __len__(self):
    return self.mask_paths.__len__() // self.batch_size


  def transforms(self):
    train_transforms = alb.Compose([
      alb.HorizontalFlip(p=0.5),
      alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0.2, shift_limit=0.2, p=0.5, border_mode=0)
    ])
    return train_transforms


  def resize(self):
    resize_transforms = alb.Resize(
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
      
      if self.aug:  # Apply augmentations.
        train_augment = self.train_transforms(image=img, mask=msk)
        img, msk = train_augment['image'], train_augment['mask']

      # Store image in x.
      self.x[j] = img / 255. # Normalizing image to be in range [0.0, 1.0]
      
      msk = self.rgb_to_onehot(msk, self.color_map, self.num_classes)
      
      self.y[j] = msk.argmax(-1)
      
    return self.x, self.y


  @classmethod
  def rgb_to_onehot(cls, rgb_arr, color_map, num_classes):
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros( shape, dtype=np.float32 )

    for i, classes in enumerate(color_map):
      arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])

    return arr

