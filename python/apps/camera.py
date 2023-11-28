# Built-in imports
import sys
import argparse
import math
# from enum import Enum
# from typing import overload, final

# Third-party imports
import numpy as np
# import cv2 as cv
import cv2
import matplotlib.pyplot as plt

# Darkest APi imports

winname = "OpenCV Window"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)



def lmb(action, x, y, flags, userdata):
  if action == cv2.EVENT_LBUTTONDOWN:
    print("left mouse button pressed")

  if action == cv2.EVENT_LBUTTONUP:
    print("left mouse button released")


def trackbar(*args):
  print(f"Trackbar: {args[0]}")





if __name__ == "__main__":
  
  source = cv2.VideoCapture(0)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  # Create trackbars
  cv2.createTrackbar("Trackbar", winname, 127, 255, trackbar)

  # Set Callbacks
  cv2.setMouseCallback(winname, lmb)

  while True:
    has_frame, frame = source.read()
    if not has_frame:	break
    frame = cv2.flip(frame, 1)



    cchar = chr(cv2.waitKey(1))
    print(cchar)
    key = cv2.waitKey(1)
    match key:
      case 99:  # c is pressed
        print(f"Key pressed: {key}")
      case 27:  # esc is pressed 
        break
      # case _: # esc is pressed 
      #   print(f"Key pressed: {key}")
    
    cv2.imshow(winname, frame)

  source.release()
  cv2.destroyAllWindows()

