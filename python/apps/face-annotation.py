from __future__ import print_function
import math

import numpy as np
import cv2 as cv



cola = (255,255,0)
colb = (0,255,255)
winname = "OpenCV Window"
cv.namedWindow(winname, cv.WINDOW_NORMAL)




def drawCircle(action, x, y, flags, userdata):
  # Referencing global variables 
  global center, circumference

  if action==cv.EVENT_LBUTTONDOWN:
    center=[(x,y)]
    cv.circle(source, center[0], 1, cola, 2, cv.LINE_AA)

  elif action==cv.EVENT_LBUTTONUP:
    circumference=[(x,y)]
    # Calculate radius of the circle
    radius = math.sqrt(
      math.pow(center[0][0]-circumference[0][0],2)
      +math.pow(center[0][1]-circumference[0][1],2)
    )
    cv.circle(source, center[0], int(radius), colb, 2, cv.LINE_AA)
    cv.imshow(winname, source)




if __name__ == '__main__':
  source = cv.imread("/home/ccpcpp/Dropbox/code/darkest/resources/images/underwater.png")
  tempsource = source.copy()

  cv.setMouseCallback(winname, drawCircle)

  k = 0
  # loop until escape character is pressed
  while k!=27 :
    cv.imshow(winname, source)
    cv.putText(
      source,
      'Choose center, and drag, Press ESC to exit and c to clear',
      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2 
    )
    k = cv.waitKey(20) & 0xFF
    if k==99:
      source = tempsource.copy()

  cv.destroyAllWindows()
