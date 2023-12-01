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



def lmb(action, x, y, flags, userdata):
  """Left mouse button event method.
  """
  if action == cv2.EVENT_LBUTTONDOWN:
    log.debug("left mouse button pressed")

  if action == cv2.EVENT_LBUTTONUP:
    log.debug("left mouse button released")


def trackbar(*args):
  log.debug(f"Trackbar: {args[0]}")




def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/images/that-space.png", help="Path to the file.")
  parser.add_argument("-wn", "--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()
  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)

  # Create trackbars
  cv2.createTrackbar("Trackbar", args.winName, 127, 255, trackbar)

  # Set Callbacks
  cv2.setMouseCallback(args.winName, lmb)

  image = cv2.imread(args.filePath)

  # key = cv2.waitKey(1)
  # match key:
  #   case 99:  # c is pressed
  #     log.debug(f"Key pressed: {key}")
  #   case 27:  # esc is pressed 
  #     break
    # case _: # esc is pressed 
    #   log.debug(f"Key pressed: {key}")
  
  cv2.imshow(args.winName, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

