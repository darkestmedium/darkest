from __future__ import print_function
import math

import numpy as np
import cv2 as cv


cola = (255,255,0)
colb = (0,255,255)
winname = "OpenCV Window"
trackbarValue = "Scale"
cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
image = cv.imread("/home/ccpcpp/Dropbox/code/darkest/resources/images/underwater.png")
maxScaleUp = 100
scaleFactor = 50

# Callback functions
def scaleImage(*args):
  scaleFactorDouble = 0.5+args[0]*0.01
  scaledImage = cv.resize(
    image, None, fx=scaleFactorDouble, fy=scaleFactorDouble, interpolation=cv.INTER_LINEAR
  )
  cv.imshow(winname, scaledImage)


if __name__ == '__main__':
  cv.createTrackbar(trackbarValue, winname, scaleFactor, maxScaleUp, scaleImage)

  cv.imshow(winname, image)
  cv.waitKey(0)
  cv.destroyAllWindows()

