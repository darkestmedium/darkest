# Built-in imports
import sys; sys.path.append("/home/ccpcpp/Dropbox/code/darkest/python")


from api import Enum
from api import overload

# Third-party imports
from api import np
from api import cv2


import api.Darkest as da
import api.DarkestUi as daui



if __name__ == "__main__":
  s = 0
  if len(sys.argv) > 1: s = sys.argv[1]
  source = cv2.VideoCapture(s)

  source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
  source.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  source.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  win_name = "Camera Preview"
  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

  net = cv2.dnn.readNetFromCaffe(
    "resources/ml/models/deploy.prototxt",
    "resources/ml/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
  )
  # Model parameters
  inwh = (256, 256)
  mean = [104, 117, 123]
  conf_threshold = 75

  prec = 0
  uidraw = daui.ocvdraw(dnn=net)
  opacity = int(0.65 * 255)

  while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:	break
    frame = cv2.flip(frame, 1)
    uidraw.image(frame)
    # Run a model
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, inwh, mean, swapRB=False, crop=False))
    detections = net.forward()
    for indx in range(detections.shape[2]):
      confidence = detections[0, 0, indx, 2] * 100
      if confidence > conf_threshold:
        uidraw.get_bbox_ss((detections[0, 0, indx, 3], detections[0, 0, indx, 4], detections[0, 0, indx, 5], detections[0, 0, indx, 6]))
        uidraw.bbox_outline(opacity=opacity)
        uidraw.text(f"person: {confidence}")


    uidraw.stats(opacity=opacity)
    cv2.imshow(win_name, uidraw.combine())

  source.release()
  cv2.destroyAllWindows()



/home/ccpcpp/Dropbox/code/resources/video/pexels-olya-kobruseva-5901087 (720p).mp4