# import numpy as np
# import tensorflow as tf
# from tensorflow import keras as tfk
# import matplotlib.pyplot as plt

# Built-in imports
import sys; sys.path.append("/home/lukas/Dropbox/code/oa/python")

from PySide6 import QtCore as qtc



# asd = qtc.QFileInfo("/Users/luky/Dropbox/dev/oa./api2.py")
# print(asd.absolutePath())


import python.api.Oa as oa


# print(oa..FileIO.listdir("/Users/luky/Dropbox/dev/oa./api"))


# print(qtc.QFileInfo("/Users/luky/Dropbox/dev/oa./api2").isDir())

# print(oa..FileIO.mkdir(qtc.QFileInfo("./asd_fi/asd.py")))
# print(oa..FileIO.mkdir(qtc.QDir("./asd_dir")))
# print(oa..FileIO.mkdir("./asd_str"))


# asd = oa..FileIO.exists('/Users/luky/Dropbox/dev/oa./api/oa.2.py')
# print(asd)
# print(oa..FileIO.exists(qtc.QFileInfo('/Users/luky/Dropbox/dev/oa./api/oa..py')))
# print(oa..FileIO.exists(qtc.QDir('/Users/luky/Dropbox/dev/oa./api')))

# print(oa..FileIO.exists(qtc.QFileInfo('/Users/luky/Dropbox/dev/oa./api/oa..py')))
# print(f"from QFileInfo {oa..FileIO.exists(16)}")

# path="/Users/luky/Dropbox/dev/test/asd/"
path="/Users/luky/Dropbox/dev/test/asd.txt"
# path="/Users/luky/Desktop/asd/sadsa/"
path = qtc.QFileInfo(path)
# qtc.QFile(path.filePath()).remove()

# if path.isFile():	qtc.QFile(path).remove()
if path.isFile():
	print("isfile")
	qtc.QFile(path.absoluteFilePath()).remove()
if path.isDir():
	print("isdir")
	path.absoluteDir().removeRecursively()
	# qtc.QDir.remove(path.absolutePath())

# list_qfi = oa..FileIO.listdir("./test/dts/101_ObjectCategories/accordion", ["*.jpg"])
# [print(qfi.filePath()) for qfi in list_qfi]
# print(list_qfi)
# ds_fasion_mnist = tfk.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = ds_fasion_mnist.load_data()

# index = 0 
# np.set_printoptions(linewidth=320)

# print("LABEL: ", train_labels[index])
# print("IMAGE PIXELS: ", train_images[index])

# plt.imshow(train_images[index], cmap="Greys")
# plt.show()

# # model = tfk.Sequential(

# # )


