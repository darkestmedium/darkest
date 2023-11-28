# Built-in imports
import argparse
import math

# Third-party imports
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

# OpenAPi imports


imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/CoinsA.png"
videoPath = "/home/ccpcpp/Dropbox/code/darkest/resources/video/focus-test.mp4"




# Implement Variance of absolute values of Laplacian 
def var_abs_laplacian(image):
  laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale=1, delta=0)
  return np.var(np.abs(laplacian-np.mean(np.abs(laplacian))))


# Implement Sum Modified Laplacian
def sum_modified_laplacian(image):
  laplacian_x = cv2.filter2D(image, -1, np.array([[0,0,0],[-1,2,-1],[0,0,0]]), (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
  laplacian_y = cv2.filter2D(image, -1, np.array([[0,-1,0],[0,2,0],[0,-1,0]]), (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
  return np.sum(np.abs(laplacian_x)+np.abs(laplacian_y))




if __name__ == "__main__":
  cap = cv2.VideoCapture(videoPath)
  ret, frame = cap.read()

  print(f"Total number of frames : {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

  max_afa = 0; max_afb = 0
  frame_a = 0; frame_b = 0 
  frame_id_a = 0; frame_id_b = 0 

  # Get measures of focus from both methods
  afa = var_abs_laplacian(frame)
  afb = sum_modified_laplacian(frame)

  # Flower ROI
  top=410; left=1050; bottom=frame.shape[0]; right=650

  # Iterate over all the frames present in the video
  while(ret):
    # Crop the flower region out of the frame
    flower = frame[left:right, top:bottom]
    # Measure af
    afa = var_abs_laplacian(frame)
    afb = sum_modified_laplacian(frame)
    
    # Check method A
    if afa > max_afa:
      max_afa = afa
      frame_id_a = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
      frame_a = frame.copy()
      print(f"Frame ID of the best frame [Method A]: {frame_id_a}")

    # Check method B
    if afb > max_afb: 
      max_afb = afb
      frame_id_b = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
      frame_b = frame.copy()
      print(f"Frame ID of the best frame [Method B]: {frame_id_b}")
      
    # Read a new frame
    ret, frame = cap.read()


  print("================================================")
  print("Frame ID of the best frame [Method A]: {}".format(frame_id_a))
  print("Frame ID of the best frame [Method B]: {}".format(frame_id_b))

  print(f"AF max value [Method A]: {max_afa}")
  print(f"AF max value [Method B]: {max_afb}")


  # Post
  cap.release()
  # Stack the best frames obtained using both methods
  out = np.hstack((frame_a, frame_b))
  # Display the stacked frames
  plt.figure()
  plt.imshow(out[:,:,::-1])
  plt.axis('off')
  plt.show()



