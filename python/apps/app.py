# Built-in imports
import argparse
import math

# Third-party imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# OpenAPi imports

imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/CoinsA.png"




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--imgpath', type=str, default='images/2.jpg', help="Image file path")


  args = parser.parse_args()
  print(args)

