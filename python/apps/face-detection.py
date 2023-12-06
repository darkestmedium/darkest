# Built-in imports
import sys; sys.path.append("/home/ccpcpp/Dropbox/code/darkest/python")

import os
import argparse
import logging

from api import Enum
from api import overload
# Third-party imports
from api import np
from api import cv2


import api.DarkestUi as daui

# Logging
log = logging.getLogger("darkest-log")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)



def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/deploy.prototxt", help="Path to the file.")
  parser.add_argument("--filePathDNN", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/res10_300x300_ssd_iter_140000_fp16.caffemodel", help="Path to the dnn file.")
  parser.add_argument("--camera", type=int, default=0, help="Index of the camera input, default is 0.")
  parser.add_argument("--width", type=int, default=1280, help="Width of the window")
  parser.add_argument("--height", type=int, default=720, help="Height of the window")
  parser.add_argument("--codec", type=str, default="MJPG", help="Stream codec")
  parser.add_argument("--mirror", type=int, default=1, help="Mirror camera's input")
  parser.add_argument("--winName", type=str, default="OpenCV Window - GTK", help="Name of the opencv window.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()
  cv2.namedWindow(args.winName, cv2.WINDOW_NORMAL)
  source = cv2.VideoCapture(args.camera)

  # Move to darkest api method.
  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(args.codec[0], args.codec[1], args.codec[2], args.codec[3]))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


  net = cv2.dnn.readNetFromCaffe(args.filePath, args.filePathDNN)

  uidraw = daui.ocvdraw(dnn=net)
  uidraw.set_style("light")
  opacity = int(0.65 * 255)

  while True:
    has_frame, frame = source.read()
    if not has_frame:	break
    frame = cv2.flip(frame, args.mirror)

    # Set the base frame for the detection drawer
    uidraw.image(frame)
  
    # Run a model
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (256, 256), [104, 117, 123], swapRB=False, crop=False))

    # Detection drawing
    detections = net.forward()
    for indx in range(detections.shape[2]):
      confidence = detections[0, 0, indx, 2] * 100
      if confidence > 75:
        dbox = daui.ocvdbox(detections[0,0,indx,3], detections[0,0,indx,4], detections[0,0,indx,5], detections[0,0,indx,6], uidraw.imwidth, uidraw.imheight)
        # Old uidrawer class var didn't have time to rewrite everything yet
        uidraw.get_bbox_ss((detections[0, 0, indx, 3], detections[0, 0, indx, 4], detections[0, 0, indx, 5], detections[0, 0, indx, 6]))

        uidraw.bbox_outline(dbox.dbox, opacity=opacity)

        # New dbox class
        uidraw.text(f"person: {confidence:.2f}", dbox.lefttop, alignh="left", alignv="above", bboxo=opacity)
        # uidraw.text(f"bottomrigh: {confidence:.2f}", dbox.rightbottom, alignh="right", alignv="below", bboxo=opacity)
        uidraw.text(f"right top", dbox.righttop, alignh="left", alignv="below", bboxo=opacity)
        uidraw.text(f"right mid", dbox.right, alignh="left", alignv="center", bboxo=opacity)
        uidraw.text(f"right bottom", dbox.rightbottom, alignh="left", alignv="above", bboxo=opacity)


    # Keyboard stuff
    key = cv2.waitKey(1)
    match key:
      case 99:  # c is pressed
        log.debug(f"Key pressed: {key}")
      case 27:  # esc is pressed 
        break

    uidraw.stats(opacity=opacity)

    cv2.imshow(args.winName, uidraw.combine())

  source.release()
  cv2.destroyAllWindows()