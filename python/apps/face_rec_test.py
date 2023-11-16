# Built-in imports
import sys; sys.path.append("/home/oa/Dropbox/code/oa/python")


from api import Enum
from api import overload

# Third-party imports
from api import np
from api import cv2


import api.Oa as oa
import api.OaUi as oaui



if __name__ == "__main__":

	s = 0
	if len(sys.argv) > 1: s = sys.argv[1]
	source = cv2.VideoCapture(s)

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
	uidraw = oaui.ocvdraw(dnn=net)
	opacity = int(0.65 * 255)

	while cv2.waitKey(1) != 27:
		has_frame, frame = source.read()
		if not has_frame:	break
		frame = cv2.flip(frame, 1)

		# uidraw = DrawOCVUi(frame, net)
		uidraw.image(frame)
		# Run a model
		net.setInput(cv2.dnn.blobFromImage(frame, 1.0, inwh, mean, swapRB=False, crop=False))
		detections = net.forward()
		for indx in range(detections.shape[2]):
			confidence = detections[0, 0, indx, 2] * 100
			# if confidence > conf_threshold:
			# 	uidraw.get_bbox_ss((detections[0, 0, indx, 3], detections[0, 0, indx, 4], detections[0, 0, indx, 5], detections[0, 0, indx, 6]))
			# 	uidraw.bbox_outline(opacity=opacity)
			# 	uidraw.text(
			# 		f"{oaui.ocvdraw.class_name}: {confidence:{prec}.{prec}f}%",
			# 		uidraw.get_bbox_pts()[0], alignh="left", alignv="above", bboxo=opacity
			# 	)
			# 	uidraw.text(
			# 		f"person: {confidence:{prec}.{prec}f}%",
			# 		uidraw.get_bbox_pts()[2], alignh="right", alignv="below", bboxo=opacity
			# 	)
			# 	uidraw.text(
			# 		f"right txt 1: {confidence:{prec}.{prec}f}%",
			# 		uidraw.get_bbox_pts()[1], alignh="left", alignv="below", bboxo=opacity
			# 	)
			# uidraw.text(
			# 	f"right txt 2: {confidence:{prec}.{prec}f}%",
			# 	uidraw.get_bbox_pts()[5], alignh="left", alignv="center", bboxo=opacity
			# )

		uidraw.stats(opacity=opacity)
		# cv2.imshow(win_name, uidraw.imcv)
		cv2.imshow(win_name, uidraw.combine())

	source.release()
	cv2.destroyAllWindows()


