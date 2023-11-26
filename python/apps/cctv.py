# Import module
import cv2 as cv
import numpy as np




cap = cv.VideoCapture("/home/ccpcpp/Dropbox/code/darkest/resources/video/chaplin.mp4")




# Main
if __name__ == '__main__':

  # Check if camera opened successfully
  if (cap.isOpened() == False):
    print("Error opening video stream or file")

  ret, frame = cap.read()

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
      cv.imshow("Video Output", frame)    
      # Wait for 25 ms before moving on to the next frame
      # This will slow down the video
      # cv.waitKey(25)
    # Break the loop
    else: 
      break


  # # Read image
  # winname = "OpenCV Window"
  # cv.namedWindow(winname, cv.WINDOW_NORMAL)
  # img = cv.imread("/home/ccpcpp/Dropbox/code/darkest/resources/images/IDCard-Satya.png")

  # qrDecoder = cv.QRCodeDetector()
  # opencvData, dbox, rectifiedImage = qrDecoder.detectAndDecode(img)

  # dbox = np.squeeze(dbox).astype(int)
  # n = dbox.__len__()
  # [cv.line(img, dbox[i], dbox[(i+1)%n], (255, 0, 255), 2) for i in range(n)]

  # print(opencvData)

  # cv.imwrite("home/ccpcpp/Dropbox/code/darkest/resources/images/QRCode-Output.png", img)

  # cv.imshow(winname, img)
  # cv.waitKey(0)