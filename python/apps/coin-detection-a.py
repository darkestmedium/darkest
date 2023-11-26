# Built-in imports
import argparse
import math

# Third-party imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['axes.facecolor'] = 'gray'

# OpenAPi imports
winname = "OpenCV Window"
cv.namedWindow(winname, cv.WINDOW_NORMAL)
imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/CoinsA.png"




def displayConnectedComponents(im):
  imLabels = im
  # The following line finds the min and max pixel values
  # and their locations in an image.
  (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(imLabels)
  # Normalize the image so the min value is 0 and max value is 255.
  imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)
  # Convert image to 8-bits unsigned type
  imLabels = np.uint8(imLabels)
  # Apply a color map
  imColorMap = cv.applyColorMap(imLabels, cv.COLORMAP_JET)
  # Display colormapped labels
  plt.imshow(imColorMap[:,:,::-1])
  plt.show()



if __name__ == "__main__":
  # Read image
  # Store it in the variable image
  ###
  ### YOUR CODE HERE
  ###
  image = cv.imread(imagePath)
  imageCopy = image.copy()
  # Convert image to grayscale
  # Store it in the variable imageGray
  ###
  ### YOUR CODE HERE
  ###
  imageGrey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  # Split cell into channels
  # Store them in variables imageB, imageG, imageR
  ###
  ### YOUR CODE HERE
  ###
  imageB, imageG, imageR = cv.split(image)
  ###
  ### YOUR CODE HERE
  ###
  thresholdedImage1 = imageG.copy()
  _, thresholdedImage1 = cv.threshold(imageG, 40, 255, cv.THRESH_BINARY)
  # Display the thresholded image
  ###
  ### YOUR CODE HERE
  ###
  plt.imshow(thresholdedImage1)
  plt.title("Tresholded Image")
  ###
  ### YOUR CODE HERE
  ###
  structElement3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
  imageEroded1 = cv.erode(thresholdedImage1, structElement3)
  plt.imshow(imageEroded1)
  plt.title("Eroded Image1")
  ###
  ### YOUR CODE HERE
  ###
  structElement7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
  imageEroded2 = cv.erode(imageEroded1, structElement7)
  plt.imshow(imageEroded2)
  plt.title("Eroded Image2")
  ###
  ### YOUR CODE HERE
  ###
  # Setup SimpleBlobDetector parameters.
  params = cv.SimpleBlobDetector_Params()
  # Change thresholds
  params.minThreshold = 10
  params.maxThreshold = 200
  # Filter by Area.
  params.minDistBetweenBlobs = 2
  params.blobColor = 0
  params.filterByArea = True
  params.minArea = 1500
  params.maxArea = 99999
  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.1
  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.87
  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.8
  # Create a detector with the parameters
  detector = cv.SimpleBlobDetector_create(params)
  # Detect blobs
  ###
  ### YOUR CODE HERE
  ###
  inverted_image = cv.bitwise_not(imageEroded2)
  keypoints = detector.detect(inverted_image)
  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # Print number of coins detected
  ###
  ### YOUR CODE HERE
  ###
  print(f"Number of coins detected: {keypoints.__len__()}")
  # Mark coins using image annotation concepts we have studied so far
  ###
  ### YOUR CODE HERE
  ###
  # cv.circle(source, center[0], 1, cola, 2, cv.LINE_AA)
    # for keypoint in keypoints:
    #   point = (int(keypoint.pt[0]), int(keypoint.pt[1]))
    #   cv.circle(image, point, int(keypoint.size), (0, 255, 0), 2, cv.LINE_AA)
    #   cv.drawMarker(image, point, (0, 255, 0), cv.MARKER_CROSS, int(keypoint.size*0.25), 2, cv.LINE_AA)
  # Find connected components
  ###
  ### YOUR CODE HERE
  ###
  th, imThresh = cv.threshold(imageEroded2, 127, 255, cv.THRESH_BINARY)
  # Find connected components
  _, imLabels = cv.connectedComponents(imThresh)
  # Print number of connected components detected
  ###
  ### YOUR CODE HERE
  ###
  print(f"Number of number of connected components detected: {imLabels.max()}")
  # Display connected components using displayConnectedComponents
  # function
  ###
  ### YOUR CODE HERE
  ###
  # displayConnectedComponents(imLabels)
  # Find all contours in the image
  ###
  ### YOUR CODE HERE
  ###
  # Find all contours in the image
  contours, hierarchy = cv.findContours(imageEroded2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
  # Print the number of contours found
  ###
  ### YOUR CODE HERE
  ###
  print(f"Number of contours found = {contours.__len__()}")
  # Draw all contours
  ###
  ### YOUR CODE HERE
  ###
  # cv.drawContours(image, contours, -1, (0,255,0), 2)
  # Remove the inner contours
  # Display the result
  ###
  ### YOUR CODE HERE
  ###
  top = int(0.025 * image.shape[0])  # shape[0] = rows
  bottom = top
  left = int(0.025 * image.shape[1])  # shape[1] = cols
  right = left
  dst = cv.copyMakeBorder(np.ones_like(imageEroded2, imageEroded2.dtype), top, bottom, left, right, cv.BORDER_CONSTANT, None, 255)
  dst = cv.resize(dst, (image.shape[1], image.shape[0]), cv.INTER_LINEAR)
  contoursOut, hierarchy = cv.findContours(dst, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
  print(f"Number of outside contours found = {contoursOut.__len__()}")
  # cv.drawContours(image, contoursOut, -1, (0,255,0), 10)
  # Print area and perimeter of all contours
  ###
  ### YOUR CODE HERE
  ###
  contours += contoursOut
  areaMax=0
  for index,cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))
    areaMax = area if area > areaMax else areaMax
  # Print maximum area of contour
  # This will be the box that we want to remove
  ###
  ### YOUR CODE HERE
  ###    
  print(f"Maximal contour area = {cv.contourArea(contours[-1])}")
  # Remove this contour and plot others
  ###
  ### YOUR CODE HERE
  ###
  # cv.drawContours(image, contours[:-1], -1, (0,255,0), 2)
  # Fit circles on coins
  ###
  ### YOUR CODE HERE
  ###
  for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    # Mark the center
    cv.drawMarker(image, (x,y), (0, 255, 0), cv.MARKER_CROSS, 10, 1, cv.LINE_AA)
    # Fit an ellipse
    # We can fit an ellipse only
    # when our contour has minimum
    # 5 points
    if len(cnt) < 5: continue
    ellipse = cv.fitEllipse(cnt)
    cv.ellipse(image, ellipse, (255,0,125), 2, cv.LINE_AA)



  # plt.show()
  cv.imshow(winname, image)
  cv.waitKey(0)
  cv.destroyAllWindows()
