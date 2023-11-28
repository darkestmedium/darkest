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

imagePath = "/home/ccpcpp/Downloads/trump.jpg"




def get_outline(image, blur=5, edge=9):
  imageblur = cv2.medianBlur(image, blur)
  return cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)




def cartoonify(image, arguments=0, blur=5, edge=9, alpha=0.5):
  ###
  ### YOUR CODE HERE
  ###
  blur=5
  edge=9
  alpha=0.5
  # Outline
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imageblur = cv2.medianBlur(imgrey, blur)
  imoutline = cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)
  # Colour quantization
  total_color = 8
  k=total_color
  data = np.float32(image).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(image.shape)
  blurred = cv2.edgePreservingFilter(result, flags=1, sigma_s=60, sigma_r=0.4)
  return cv2.bitwise_and(blurred, blurred, mask=imoutline)




def pencilSketch(image, arguments=0, blur=5, edge=9, alpha=0.5):
  ###
  ### YOUR CODE HERE
  ###
  blur=5
  edge=9
  alpha=0.5
  imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imageblur = cv2.medianBlur(imgrey, blur)
  imoutline = cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)
  pencilSketchImage = cv2.addWeighted(imgrey, 1-alpha, imoutline, alpha, 0)
  return imout




# def spiderVerse(image, blur=5, edge=9, alpha=0.5, offset_r=5, offset_b=-5):
#   imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   imageblur = cv2.medianBlur(imgrey, blur)
#   imoutline = cv2.adaptiveThreshold(imageblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge, blur)

#   # Split the image into color channels
#   b, g, r = cv2.split(image)

#   # Create red and blue channel offsets
#   rows, cols = image.shape[:2]
#   offset_matrix_r = np.roll(np.eye(rows, cols), offset_r, axis=(0, 1))
#   offset_matrix_b = np.roll(np.eye(rows, cols), offset_b, axis=(0, 1))

#   # Apply the offsets to the red and blue channels
#   # r_offset = cv2.warpAffine(r, offset_matrix_r, (cols, rows))
#   # b_offset = cv2.warpAffine(b, offset_matrix_b, (cols, rows))

#   # Merge the channels back together
#   distorted_img = cv2.merge([offset_matrix_b, g, offset_matrix_r])

#   return distorted_img





if __name__ == "__main__":
  image = cv2.imread(imagePath)


  # cv2.imshow(winname, pencilSketch(image))
  cv2.imshow(winname, cartoonify(image))


  cv2.waitKey(0)
  cv2.destroyAllWindows()

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

