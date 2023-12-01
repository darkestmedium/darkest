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


log = logging.getLogger("darkest")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)

filePath = f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/images/scan_out.png"



data = {
  "imgout": None,
}




def lmb(action, x, y, flags, userdata):
  """Left mouse button event method.
  """
  if action == cv2.EVENT_LBUTTONDOWN:
    log.debug(f"LMB pressed at: {x} x {y}")

  if action == cv2.EVENT_LBUTTONUP:
    log.debug(f"LMB released at: {x} x {y}")

    cv2.imwrite(filePath, data["imgout"])
    log.debug(f"Image saved: {filePath}")


def trackbar(*args):
  value = args[0]
  if value > 0:
    # using thresholding on warped image to get scanned effect (If Required)
    ret, th1 = cv2.threshold(data["imgout"], 127,255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(data["imgout"], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(data["imgout"], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret2, th4 = cv2.threshold(data["imgout"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data["imgout"] = th4

    log.debug(f"Image Enhanced: {args[0]}")



def unwarp(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2), dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]

  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew


def get_contours(points):
  # find the contours in the edged image, keeping only the
  # largest ones, and initialize the screen contour
  contours, _ = cv2.findContours(points, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)

  # get approximate contour
  for contour in contours:
    pt = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * pt, True)
    if approx.__len__() == 4:
      target = approx
      break

  # mapping target points to 800x800 quadrilateral
  return (unwarp(target), target)


def get_resolution(contours, format:str="a4"):
  """Gets the dimensions and orientation of the document.
  """
  min_x, min_y = np.min(contours, axis=0)
  max_x, max_y = np.max(contours, axis=0)
  # Calculate width and height
  width = int(max_x - min_x)
  height = int(max_y - min_y)

  mode = "portrait"
  if width < height:
    mode = "portrait"
    log.debug("Mode: Portrait")
  if width > height:
    mode = "landscape"
    log.debug("Mode: Landscape")

  if format == "a4":
    if mode == "portrait":
      width_mm = 210
      height_mm = 297
    if mode == "landscape":
      width_mm = 297
      height_mm = 210
    

  aspect_ratio = (width_mm / height_mm)
  if width / height > aspect_ratio: height = int(width / aspect_ratio)
  else: width = int(height * aspect_ratio)

  return (width, height)





def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/images/scanned-form.jpg", help="Path to the file.")
  parser.add_argument("-ar", "--aspectRatio", type=str, default=f"a4", help="Format Aspect Ratio")
  parser.add_argument("-wn", "--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()
  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)

  # Create trackbars
  cv2.createTrackbar("Trackbar", args.winName, 0, 1, trackbar)
  # Set Callbacks
  cv2.setMouseCallback(args.winName, lmb)

  image = cv2.imread(args.filePath)
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imgblur = cv2.medianBlur(imgrey, 7)
  imgedges = cv2.Canny(imgblur, 0, 50)


  approx, target = get_contours(imgedges)
  width, height = get_resolution(approx)

  persptrans = cv2.getPerspectiveTransform(approx, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
  imgout = cv2.warpPerspective(image, persptrans, (width, height))

  cv2.drawContours(image, [target], -1, (0, 255, 0), 2)

  data["imgout"] = cv2.cvtColor(imgout, cv2.COLOR_BGR2GRAY)

  # key = cv2.waitKey(1)
  # match key:
  #   case 99:  # c is pressed
  #     log.debug(f"Key pressed: {key}")
  #   case 27:  # esc is pressed 
  #     break
  #   case _: # esc is pressed 
  #     log.debug(f"Key pressed: {key}")
  
  cv2.imshow(args.winName, image)

  cv2.waitKey(0)
  cv2.destroyAllWindows()