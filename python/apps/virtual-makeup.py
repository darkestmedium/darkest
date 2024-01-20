# Built-in imports
import sys
import os
import argparse
import logging
from typing import Iterable

# Third-party imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from mediapipe.python.solutions.face_mesh import FaceMesh


# Darkest APi imports



log = logging.getLogger("darkest-log")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)




data = {
  "image": None,
  "temp": None,
  "output": None,
  # Landmarks
  "lipUpper": [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76],
  "lipLower": [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  "connFace": [10, 338, 297, 332, 284, 251, 389, 264, 447, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 138, 213, 147, 234, 127, 162, 21, 54, 103, 67, 109],
  "cheeks": [425, 205]
  # Colors
}





class Landmarks():
  """Namespace class for landmarks.
  """


  @classmethod
  def detect(cls, src:np.ndarray, isStream:bool=False):
    """Detects landmarks for the given image.
    """
    with FaceMesh(static_image_mode=not isStream, max_num_faces=1) as face_mesh:
      results = face_mesh.process(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
      return results.multi_face_landmarks[0].landmark
    return None


  @classmethod
  def normalize(cls, landmarks, height:int, width:int, mask:Iterable=None):
    """The landmarks returned by mediapipe have coordinates between [0, 1].
    This function normalizes them in the range of the image dimensions so they can be played with.
    """
    normalized_landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])
    if mask: normalized_landmarks = normalized_landmarks[mask]
    return normalized_landmarks


  @classmethod
  def plot(cls, src:np.array, landmarks:list):
    """Given a source image and a list of landmarks plots them onto the image
    """
    dst = src.copy()
    [cv.circle(dst, (x, y), 2, 0, cv.FILLED) for x,y in landmarks]
    return dst




class Mask():
  """Namespace for masks.
  """


  @classmethod
  def lips(cls, src:np.ndarray, points:np.ndarray, color:list) -> np.ndarray:
    """Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)
    mask = cv.fillPoly(mask, [points], color)
    return cv.GaussianBlur(mask, (7, 7), 5)


  @classmethod
  def blush(cls, src:np.ndarray, points:np.ndarray, color:list, radius:int) -> np.ndarray:
    """Given a src image, points of the cheeks, desired color and radius
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)
    for point in points:
      mask = cv.circle(mask, point, radius, color, cv.FILLED)
      x, y = point[0]-radius, point[1]-radius
      mask[y:y+2*radius, x:x+2*radius] = cls.vignette(mask[y:y+2*radius, x:x+2*radius], 10)
    return mask


  @classmethod
  def vignette(cls, src:np.ndarray, sigma:int):
    """Given a src image and a sigma, returns a vignette of the src.
    """
    height, width, _ = src.shape
    kernel_x = cv.getGaussianKernel(width, sigma)
    kernel_y = cv.getGaussianKernel(height, sigma)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    blurred = cv.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    return blurred




def lmb(action, x, y, flags, userdata):
  """Left mouse button event method.
  """
  if action == cv.EVENT_LBUTTONDOWN:
    log.debug("left mouse button pressed")

  if action == cv.EVENT_LBUTTONUP:
    log.debug("left mouse button released")


def blendResults(*args):
  alpha = args[0]*0.01
  beta = (1.0 - alpha)
  data["output"] = cv.addWeighted(data["temp"], alpha, data["image"], beta, 0.0)

  log.debug(f"Trackbar: {alpha}")




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/resources/images/girl-no-makeup.jpg", help="Path to the file.")
  parser.add_argument("-wn", "--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  parser.add_argument("-al", "--applyLipstick", type=bool, default=True, help="Whether or not to apply the lipstick.")
  parser.add_argument("-ab", "--applyBlush", type=bool, default=True, help="Whether or not to apply the blush.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntaxCreator()
  cv.namedWindow(args.winName, cv.WINDOW_NORMAL)

  data["image"] = cv.imread(args.filePath, cv.IMREAD_COLOR)
  data["temp"] = data["image"]
  data["output"] = data["image"]

  # Set Callbacks
  # cv.setMouseCallback(args.winName, lmb)

  height, width = data["image"].shape[:-1]
  landmarks = Landmarks.detect(data["image"])

  landmarksFeature = None
  if args.applyLipstick:
    landmarksFeature = Landmarks.normalize(landmarks, height, width, data["lipLower"]+data["lipUpper"])
    mask = Mask.lips(data["image"], landmarksFeature, [153, 0, 157])
    data["temp"] = cv.addWeighted(data["image"], 1.0, mask, 0.4, 0.0)
    log.debug("Apply lipstick")

  if args.applyBlush:
    landmarksFeature = Landmarks.normalize(landmarks, height, width, data["cheeks"])
    mask = Mask.blush(data["image"], landmarksFeature, [153, 0, 157], 50)
    data["temp"] = cv.addWeighted(data["temp"], 1.0, mask, 0.3, 0.0)
    log.debug("Apply blush")

  # Create trackbars
  cv.createTrackbar("Blend Results", args.winName, 100, 100, blendResults)

  k=0
  while k!=27:
    cv.imshow(args.winName, data["output"])
    cv.waitKey(1)

  cv.destroyAllWindows()

