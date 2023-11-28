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
imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/blemish.png"




"""
Description

In this project, you will implement any algorithm of your choice for chroma-keying. We have linked to a few resources and pieces of code you can look at.

Input: The input to the algorithm will be a video with a subject in front of a green screen.
Output: The output should be another video where the green background is replaced with an interesting background of your choice. The new background could even be a video if you want to make it interesting.

Controls: You can build a simple interface using HighGUI. It should contain the following parts.
  Color Patch Selector: The interface should show a video frame and the user should be allowed to select a patch of green screen from the background. For simplicity, this patch can be a rectangular patch selected from a single frame. However, it is perfectly find to build an interface where you select multiple patches from one or more frames of the video.
  Tolerance slider: This slider will control how different the color can be from the mean of colors sampled in the previous step to be included in the green background.
  Softness slider (Optional): This slider will control the softness of the foreground mask at the edges.
  Color cast slider (Optional): In a green screen environment, some of the green color also gets cast on the subject. There are some interesting ways the color cast can be reduced, but during the process of removing the color cast some artifacts are get introduced. So, this slider should control the amount of color cast removal we want.

References:
  Blue Screen Matting : This paper is a pioneering paper by Alvy Ray Smith and James Blinn. Alvy Ray Smith was a co-founder of Pixar, and winner of two technical oscar awards.
  A C language implementation
  Robust Chroma Keying System based on Human Visual Perception and Statistical Color Models
  State of the Art (non-realtime) by Disney. Also checkout the video.
"""




def color_picker(action, x, y, flags, userdata):

  if action == cv2.EVENT_LBUTTONDOWN:
    print("left mouse button pressed")
    # get upper threshold


  if action == cv2.EVENT_LBUTTONUP:
    print("left mouse button released")
    # get lower threshold





if __name__ == "__main__":
  source = cv2.VideoCapture(0)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  # Set Callbacks
  cv2.setMouseCallback(winname, color_picker)


  key = 0
  while True:
    has_frame, frame = source.read()
    frame = cv2.flip(frame, 1)

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
