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




data = {
  "image": None,
  # "cola": [0,255,0],  # darker
  # "colb": [0,255,0],  # brighter
  "colop": [87, 90, 88],  # darker
  "color": [106, 107, 105], # brighter
}



def get_luminance(color):
  return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]


def lmb(action, x, y, flags, userdata):
  """Left mouse button event method.
  """

  if action == cv2.EVENT_LBUTTONDOWN:
    data["colop"] = data["image"][y, x]
    print(f"Sampled color on press:   {data['colop']} at {x} x {y}")

  if action == cv2.EVENT_LBUTTONUP:
    data["color"] = data["image"][y, x]
    print(f"Sampled color on release: {data['color']} at {x} x {y}")

  # Sort colors - compare the luminance of pressed and released colors
  lumiop = get_luminance(data["colop"])
  lumior = get_luminance(data["color"])
  if lumiop < lumior:
    data["colop"] = data["colop"]
    data["color"] = data["color"]
  elif lumiop > lumior:
    data["colop"] = data["color"]
    data["color"] = data["colop"]
  else:
    data["colop"] = data["colop"]
    data["color"] = data["colop"]


def tolerance(*args):
  print(f"tolerance: {args[0]}")


def softness(*args):
  print(f"softness: {args[0]}")


def defringe(*args):
  print(f"defringe: {args[0]}")




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--imagePath", type=str, default="/home/ccpcpp/Dropbox/code/darkest/resources/images/that-space.png", help="Image file path.")
  parser.add_argument("--width", type=int, default=1280, help="Width of the window")
  parser.add_argument("--height", type=int, default=720, help="Height of the window")
  parser.add_argument("--camera", type=int, default=0, help="Index of the camera input, default is 0.")
  parser.add_argument("--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")

  return parser.parse_args()



if __name__ == "__main__":

  args = syntaxCreator()

  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)
  source = cv2.VideoCapture(args.camera)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

  # Create trackbars
  # cv2.createTrackbar("Tolerance", args.winName, 0, 255, tolerance)
  cv2.createTrackbar("Softness", args.winName, 0, 255, tolerance)
  cv2.createTrackbar("Defringe", args.winName, 0, 255, tolerance)

  # Set Callbacks
  cv2.setMouseCallback(args.winName, lmb)


  while True:
    _, image = source.read()
    image = cv2.flip(image, 1)
    data["image"] = image  # Pass data to callback functions

    imback = cv2.imread(args.imagePath)
    imback = cv2.resize(imback, (args.width, args.height))
    # crop_background = imback[0:args.height, 0:args.width]

  
    lower = np.array(data["colop"]).astype(np.uint8)
    upper = np.array(data["color"]).astype(np.uint8)
    mask = cv2.inRange(image, lower, upper)
    # res = cv2.bitwise_and(image, image, mask = mask)

    masked_image = np.copy(image)

    masked_image[mask != 0] = [0, 0, 0]
    imback[mask == 0] = [0, 0, 0]

    imout = imback + masked_image
  

    key = cv2.waitKey(1)
    match key:
      case 99:  # c is pressed
        print(f"Key pressed: {key}")
      case 27:  # esc is pressed 
        break
      # case _: # esc is pressed 
      #   print(f"Key pressed: {key}")

    cv2.imshow(args.winName, imout)

  source.release()
  cv2.destroyAllWindows()
