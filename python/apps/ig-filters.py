# Built-in imports
import sys
import argparse
import math

# Third-party imports
import numpy as np
# import cv2 as cv
import cv2
import matplotlib.pyplot as plt

# Darkest APi imports
winname = "OpenCV Window"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/trump.jpg"


import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'




def get_outline(image, blur=5, edge=9):
  imageblur = cv2.medianBlur(image, blur)
  return cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)


def color_quantize(image, colors=8):
  """Colour quantization.
  """
  data = np.float32(image).reshape((-1, 3))
  print(data)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
  ret, label, center = cv2.kmeans(data, colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  return result.reshape(image.shape)




def cartoonify(image, arguments=0, blur=5, edge=9, alpha=0.5):
  ###
  ### YOUR CODE HERE
  ###
  # Outline
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imoutline = get_outline(imgrey)
  # Colour quantization
  imgquantized = color_quantize(image)
  blurred = cv2.edgePreservingFilter(imgquantized, flags=1, sigma_s=60, sigma_r=0.4)
  cartoonImage = cv2.bitwise_and(blurred, blurred, mask=imoutline)
  return cartoonImage




def pencilSketch(image, arguments=0, blur=5, edge=9, alpha=0.5):
  ###
  ### YOUR CODE HERE
  ###
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imgreyquant = color_quantize(imgrey)
  imageblur = cv2.medianBlur(imgreyquant, blur)
  imoutline, _ = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.075, shade_factor=0.085)
  pencilSketchImage = cv2.addWeighted(imageblur, 1-alpha, imoutline, alpha, 0)
  pencilSketchImage = cv2.cvtColor(pencilSketchImage, cv2.COLOR_GRAY2BGR)
  return pencilSketchImage




def dmt(image, blur=5, edge=9, alpha=0.5, offset_r=5, offset_b=-5):
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imgreyquant = color_quantize(imgrey)
  imageblur = cv2.medianBlur(imgreyquant, blur)
  imoutline = cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)

  # Split the image into color channels
  b, g, r = cv2.split(image)


  return imoutline*imageblur





if __name__ == "__main__":
  image = cv2.imread(imagePath)


  # cv2.imshow(winname, pencilSketch(image))
  # # cv2.imshow(winname, cartoonify(image))
  # # cv2.imshow(winname, dmt(image))
  cartoonImage = cartoonify(image)
  pencilSketchImage = pencilSketch(image)

  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  plt.figure(figsize=[20,10])
  plt.subplot(131);plt.imshow(image[:,:,::-1])
  plt.subplot(132);plt.imshow(cartoonImage[:,:,::-1])
  plt.subplot(133);plt.imshow(pencilSketchImage[:,:,::-1])

  # s = 0
  # if len(sys.argv) > 1: s = sys.argv[1]
  # source = cv2.VideoCapture(s)

  # source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  # source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  # source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  # while cv2.waitKey(1) != 27:
  #   has_frame, frame = source.read()
  #   if not has_frame:	break
  #   frame = cv2.flip(frame, 1)

  #   # imout = cv2.edgePreservingFilter(frame, flags=cv2.RECURS_FILTER)
  #   # imout = cartoonify(frame)
  #   imout = pencilSketch(frame)

  #   cv2.imshow(winname, imout)

  # source.release()
  # cv2.destroyAllWindows()

