# Built-in imports
import sys
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


log = logging.getLogger("chroma-keying")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)



data = {
  "image": None,
  "colop": [87, 90, 88],  # darker
  "color": [106, 107, 105], # brighter
  "softness": 0
}



def get_luminance(color) -> float:
  return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]


def sort_colors():
  """Sorts the colors by comparing the luminance of pressed and released colors
  """
  lumiop = get_luminance(data["colop"])
  lumior = get_luminance(data["color"])
  if lumiop > lumior:
    data["colop"] = data["color"]
    data["color"] = data["colop"]
  else:
    data["colop"] = data["colop"]
    data["color"] = data["color"]


def lmb(action, x, y, flags, userdata):
  """Left mouse button event method.
  """
  if action == cv2.EVENT_LBUTTONDOWN:
    data["colop"] = data["image"][y, x]
    log.debug(f"Sampled color on press:   {data['colop']} at pixel {x} x {y}")

  if action == cv2.EVENT_LBUTTONUP:
    data["color"] = data["image"][y, x]
    log.debug(f"Sampled color on release: {data['color']} at pixel {x} x {y}")

  sort_colors()


def softness(*args):
  data["softness"] = args[0]


def defringe(*args):
  print(f"defringe: {args[0]}")




def syntax_creator():
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
  args = syntax_creator()
  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)
  source = cv2.VideoCapture(args.camera)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

  # Create trackbars
  cv2.createTrackbar("Softness", args.winName, 0, 100, softness)
  # cv2.createTrackbar("Defringe", args.winName, 0, 255, defringe)

  # Set Callbacks
  cv2.setMouseCallback(args.winName, lmb)


  while True:
    _, image = source.read()
    image = cv2.flip(image, 1)
    data["image"] = image  # Pass data to callback functions

    imgback = cv2.imread(args.imagePath)
    imgback = cv2.resize(imgback, (args.width, args.height))  # resize

    mask = cv2.inRange(
      image, 
      np.array(data["colop"]).astype(np.uint8),
      np.array(data["color"]).astype(np.uint8)
    )
    # There are prolly better ways of doing it but sometimes the simplest ones are the best ^.^
    # "It works on my machine"
    softness = data["softness"]
    if softness > 0: mask = cv2.blur(mask, (softness, softness))

    image[mask!=0] = [0, 0, 0]
    imgback[mask==0] = [0, 0, 0]

    # Defringe
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # a_channel = lab[:,:,1]
    # th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # # masked = cv2.bitwise_and(image, image, mask = th)    # contains dark background
    # m1 = image.copy()
    # m1[th==0]=(255,255,255) 
    # mlab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # dst = cv2.normalize(mlab[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # threshold_value = 100
    # dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    # mlab2 = mlab.copy()
    # mlab[:,:,1][dst_th == 255] = 127
    # img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    # img2[th==0]=(255,255,255)

    # imgout = imgback + image
    imgout = imgback + image
  
    key = cv2.waitKey(1)
    match key:
      case 99:  # c is pressed
        print(f"Key pressed: {key}")
      case 27:  # esc is pressed 
        break

    cv2.imshow(args.winName, imgout)

  source.release()
  cv2.destroyAllWindows()
