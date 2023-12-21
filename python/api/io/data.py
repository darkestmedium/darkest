# Built-in imports
import logging
import pickle
import subprocess
import platform
from typing import overload, Union, Sequence

# Third-party imports
from api import qtc
from api import tfk




class DataIO():
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