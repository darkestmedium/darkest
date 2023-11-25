from __future__ import print_function
import numpy as np
import cv2 as cv


# Main
if __name__ == '__main__':
  # Read image
  winname = "OpenCV Window"
  cv.namedWindow(winname, cv.WINDOW_NORMAL)
  img = cv.imread("/home/ccpcpp/Dropbox/code/darkest/resources/images/IDCard-Satya.png")

  qrDecoder = cv.QRCodeDetector()
  opencvData, dbox, rectifiedImage = qrDecoder.detectAndDecode(img)

  dbox = np.squeeze(dbox).astype(int)
  n = dbox.__len__()
  [cv.line(img, dbox[i], dbox[(i+1)%n], (255, 0, 255), 2) for i in range(n)]

  print(opencvData)

  cv.imwrite("home/ccpcpp/Dropbox/code/darkest/resources/images/QRCode-Output.png", img)

  cv.imshow(winname, img)
  cv.waitKey(0)
