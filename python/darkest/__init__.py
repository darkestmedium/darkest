# Built-in imports
import sys
import os
import math
import logging

import subprocess as subp
import platform
from typing import overload, final
from typing import Union, Sequence
from enum import Enum

# Third-party imports
# import numpy as np
# import cv2 as cv
# import tensorflow as tf
# from tensorflow import keras as tfk
# import albumentations as alb

# import PySide6.QtCore as qtc

# Open APi imports
# import darkest.Core
# import darkest.Ml
# import python.darkest.Ui


# # Resources file paths
# # try:
# fihome = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.HomeLocation)
# fidesktop = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DesktopLocation)
# fidownload = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DownloadLocation)
# fidocuments = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DocumentsLocation)
# # Integrate into open api? will be heavy but it is a crucial dependency
# firesources = qtc.QFileInfo("resources/")
# fids = qtc.QFileInfo("resources/ds/")
# fiml = qtc.QFileInfo("resources/ml/")
# fimlmodels = qtc.QFileInfo("resources/ml/models/")


# logging
log = logging.getLogger("darkest-api")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)



""" Darkest APi


Label imports:
	# Built-in imports:
		import subprocess as subp

	# Third-party imports:
		import numpy as np
		import PySide6.QtCore as qtc

	# Open APi imports:
		import api.Open as api
		import api.OpenUi as apiui

		import api.Open as oap
		import api.OpenUi as oapui


Naming:
	api.module.classfile.ClassName.method_name()

Open:
	api.io.file.FileIO.mkdir() -> api.io.file.iofile(FileIO).mkdir()
	api.io.data.DataIO.mkdir() -> api.io.data.iodata(DataIO).mkdir()
	import api.Open as oap
	oap.iofile.mkdir(path)

OpenUi:
	api.ui.ocvui.DrawOCVUi.draw() -> api.ui.ocvui.uicvdraw(DrawOCVUi).draw()
	import api.OpenUi as oapui
	oapui.ocvdraw.draw()


"""