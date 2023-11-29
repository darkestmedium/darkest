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



log = logging.getLogger("darkest-log")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)




def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--imagePath", type=str, default="/home/ccpcpp/Dropbox/code/darkest/resources/images/emir.jpg", help="Image file path.")
  parser.add_argument("--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  return parser.parse_args()


import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'



if __name__ == "__main__":

  args = syntax_creator()
  # cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)

  # Read 8-bit color image.
  # This is an image in which the three channels are
  # concatenated vertically.

  ###
  ### Read Image
  ###
  image = cv2.imread(args.imagePath, cv2.IMREAD_GRAYSCALE)
  # Find the width and height of the color image
  height, width = (image.shape[0]//3, image.shape[1])
  log.info(f"Image size: {height}, {width}")

  # Extract the three channels from the gray scale image
  # and merge the three channels into one color image
  imgcol = np.zeros((height, width, 3), dtype=np.uint8)
  for idx in range(3): imgcol[:,:,idx] = image[idx*height:(idx+1)*height,:]
  
  blue, green, red = (imgcol[:,:,0], imgcol[:,:,1], imgcol[:,:,2])

  # plt.figure(figsize=(20,12))
  # plt.subplot(1,3,1)
  # plt.imshow(blue)
  # plt.subplot(1,3,2)
  # plt.imshow(green)
  # plt.subplot(1,3,3)
  # plt.imshow(red)
  # plt.show()


  ###
  ### Detect Features
  ###
  MAX_FEATURES = 2048
  GOOD_MATCH_PERCENT = 0.1
  orb = cv2.ORB_create(MAX_FEATURES)
  keyptsB, descriptorsB = orb.detectAndCompute(blue, None)
  keyptsR, descriptorsR = orb.detectAndCompute(red, None)
  keyptsG, descriptorsG = orb.detectAndCompute(green, None)


  # plt.figure(figsize=[20,10])
  # img2 = cv2.drawKeypoints(blue, keyptsB, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # plt.subplot(131);plt.imshow(img2[...,::-1])

  # img2 = cv2.drawKeypoints(green, keyptsG, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # plt.subplot(132);plt.imshow(img2[...,::-1])

  # img2 = cv2.drawKeypoints(red, keyptsR, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # plt.subplot(133);plt.imshow(img2[...,::-1])


  ###
  ### Match Features
  ###
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

  # Match features between blue and Green channels
  matchesBG = list(matcher.match(descriptorsB, descriptorsG, None))
  # Sort matches by score
  matchesBG.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matchesBG) * GOOD_MATCH_PERCENT)
  matchesBG = matchesBG[:numGoodMatches]

  # Draw top matches
  imMatchesBlueGreen = cv2.drawMatches(blue, keyptsB, green, keyptsG, matchesBG, None)

  plt.figure(figsize=(12,12))
  plt.imshow(imMatchesBlueGreen[:,:,::-1])
  plt.show()


  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

  # Match features between Red and Green channels
  matchesRG = list(matcher.match(descriptorsR, descriptorsG, None))

  # Sort matches by score
  matchesRG.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matchesRG) * GOOD_MATCH_PERCENT)
  matchesRG = matchesRG[:numGoodMatches]

  # Draw top matches
  imMatchesRedGreen = cv2.drawMatches(red, keyptsR, green, keyptsG, matchesRG, None)

  plt.figure(figsize=(12,12))
  plt.imshow(imMatchesRedGreen[:,:,::-1])
  plt.show()


  ###
  ### Calculate Homography
  ###
  # Extract location of good matches - Blue -> Green
  points1 = np.zeros((len(matchesBG), 2), dtype=np.float32)
  points2 = np.zeros((len(matchesBG), 2), dtype=np.float32)

  for idx, match in enumerate(matchesBG):
    points1[idx, :] = keyptsB[match.queryIdx].pt
    points2[idx, :] = keyptsG[match.trainIdx].pt
  # Find homography
  hBlueGreen, maskBlueGreen = cv2.findHomography(points1, points2, cv2.RANSAC)


  # Extract location of good matches Gree -> Red
  points1 = np.zeros((len(matchesRG), 2), dtype=np.float32)
  points2 = np.zeros((len(matchesRG), 2), dtype=np.float32)

  for idx, match in enumerate(matchesRG):
    points1[idx, :] = keyptsR[match.queryIdx].pt
    points2[idx, :] = keyptsG[match.trainIdx].pt

  # Find homography
  hRedGreen, maskRedGreen = cv2.findHomography(points1, points2, cv2.RANSAC)


  ###
  ### Warping Image
  ###
  # Use homography to find blueWarped and RedWarped images
  warpedB = cv2.warpPerspective(blue, hBlueGreen, (width, height))
  warpedR = cv2.warpPerspective(red, hRedGreen, (width, height))

  plt.figure(figsize=(20,10))
  plt.subplot(121);plt.imshow(warpedB);plt.title("Blue channel aligned w.r.t green channel")
  plt.subplot(122);plt.imshow(warpedR);plt.title("Red channel aligned w.r.t green channel")


  ###
  ### Merge Channels
  ###
  colorImage = cv2.merge((warpedB, green, warpedR))
  originalImage = cv2.merge((blue, green, red))

  plt.figure(figsize=(20,10))
  plt.subplot(121);plt.imshow(originalImage[:,:,::-1]);plt.title("Original Mis-aligned Image")
  plt.subplot(122);plt.imshow(colorImage[:,:,::-1]);plt.title("Aligned Image")




  # imgout = image



  # cv2.imshow(args.winName, imgout)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  plt.show()