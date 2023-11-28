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





if __name__ == "__main__":
  source = cv2.VideoCapture(0)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


  key = 0
  while True:
    has_frame, frame = source.read()
    if not has_frame:	break
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