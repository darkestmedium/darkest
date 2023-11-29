# Built-in imports
import sys
import os
import argparse
import math
import logging
# from enum import Enum
# from typing import overload, final

# Third-party imports
import numpy as np
# import cv2 as cv
import cv2
import matplotlib.pyplot as plt

# Darkest APi imports


log = logging.getLogger("darkest-log")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)



import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'



def stitch_images(images):
  stitcher = cv2.Stitcher_create()
  status, panorama = stitcher.stitch(images)
  if status == cv2.Stitcher_OK:
    return panorama
  else:
    print(f"Panorama stitching failed with error code {status}")
    return None




def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--imagePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources", help="Image file path.")
  parser.add_argument("--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  return parser.parse_args()





if __name__ == "__main__":
  args = syntax_creator()
  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)

  images = []
  dirName = "scene"
  filePath = f"{args.imagePath}/images/{dirName}"
  imagefiles = [f"{filePath}/{img}" for img in os.listdir(filePath) if img.endswith(".jpg")]
  imagefiles.sort()

  destination = "{}_result.png".format(dirName)
  plt.figure(figsize=[20,15])
  i=1
  [images.append(cv2.imread(filename)) for filename in imagefiles]

  panorama = stitch_images(images)



  # image = cv2.imread(args.imagePath)
  cv2.imshow(args.winName, panorama)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

