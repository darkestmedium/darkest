# Built-in imports
import os
import argparse
import math

# Third-party imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Darkest APi imports

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/CoinsA.png"



def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--imagePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/images/that-space.png", help="Image file path.")
  parser.add_argument("--width", type=int, default=1280, help="Width of the window")
  parser.add_argument("--height", type=int, default=720, help="Height of the window")
  parser.add_argument("--camera", type=int, default=0, help="Index of the camera input, default is 0.")
  parser.add_argument("--winName", type=str, default="OpenCV Window", help="Name of the opencv window.")
  return parser.parse_args()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--imgpath', type=str, default='images/2.jpg', help="Image file path.")
  args = parser.parse_args()
  print(args)




def convertBGRtoGray(image):
  if image.shape.__len__() != 3 or image.shape[2] != 3:
    raise ValueError("Input must be a BGR image with shape (height, width, 3)")
  img_grey = np.dot(image, np.array([0.114, 0.587, 0.299]))
  # Ensure the resulting image is in the correct shape (height, width)
  if len(img_grey.shape) == 3: img_grey = img_grey[:,:,0]

  return img_grey.astype(np.uint8)


def bgr_to_hsv_from_scratch(image):
  if image.shape.__len__() != 3 or image.shape[2] != 3:
    raise ValueError("Input must be a BGR image with shape (height, width, 3)")

  image_normalized = image / 255.0

  # Extract the B, G, and R channels
  b, g, r = np.split(image_normalized, 3, axis=2)
  b = b.reshape((image.shape[0], image.shape[1]))
  g = g.reshape((image.shape[0], image.shape[1]))
  r = r.reshape((image.shape[0], image.shape[1]))

  v = np.maximum(np.maximum(r, g), b)
  s = np.where(v == 0, 0, (v-np.minimum(np.minimum(r, g), b)) / v)

  # Calculate Hue (H)
  h = np.where(v == r, (g-b) / (v-np.minimum(b, g)), np.where(v == g, 2.0 + (b-r) / (v-np.minimum(b, r)), 4.0 + (r-g) / (v-np.minimum(g, r))))
  h = (h/6.0) % 1.0

  # Scale Hue to the range [0, 179]
  h *= 179

  # Combine H, S, and V channels
  hsv_image = np.stack((h, s * 255, v * 255), axis=2)

  return hsv_image.astype(np.uint8)
