# Built-in imports
# import json
import sys; sys.path.append("/Users/luky/Dropbox/code/oa/python")

# Third-party imports
from api import np
from api import qtc
from api import cv2

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk
from PySide6 import QtCore as qtc
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Open APi imports
from api import fimlmodels
import api.Oa as oa


(x_train, _), (x_test, _) = tfk.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# print (x_train.shape)
# print (x_test.shape)




@tfk.saving.register_keras_serializable()
class DNN(tfk.Model):
	"""Keras Model subclass wrapper with convinience methods.
	"""
	fimodel = ...
	fimodelcp = ...

	# # Constructor
	# def __init__(self, latent_dim, shape, name:str="DNN", **kwargs):
	# 	super(DNN, self).__init__(name=name, **kwargs)
	# 	self.fimodel(qtc.QFileInfo(f"/Users/luky/Dropbox/dev/dnn/models/{name}/"))
	# 	self.fimodelcp(qtc.QFileInfo(f"/Users/luky/Dropbox/dev/dnn/models/{name}/checkpoint/"))
	# 	[ocv.FileIO.mkdir(fi.absolueFilePath()) for fi in [self.fimodel, self.fimodelcp]]


	@classmethod
	def cb_checkpoint(cls, path:str=None, best:bool=True):
		"""Call back for saving the best model during training.
		"""
		if not path: path = cls.fimodelcp.absoluteFilePath()
		oa.iofile.mkdir(path)
		return tf.keras.callbacks.ModelCheckpoint(
			filepath=path,
			monitor="accuracy",
			mode="max",
			save_best_only=best,
			verbose=1,
		)




@tfk.saving.register_keras_serializable()
class ReconstructAE(DNN):
	fimodel = ...
	fimodelcp = ...

	def __init__(self, latent_dim, shape, name:str="tfk_ae_reconstruct", **kwargs):
		super(ReconstructAE, self).__init__(name=name, **kwargs)
		ReconstructAE.fimodel = qtc.QFileInfo(f"{fimlmodels.absoluteFilePath()}{name}/")
		ReconstructAE.fimodelcp = qtc.QFileInfo(f"{ReconstructAE.fimodel.absoluteFilePath()}checkpoint/")
		[oa.iofile.mkdir(fi.absoluteFilePath()) for fi in [ReconstructAE.fimodel, ReconstructAE.fimodelcp]]

		self.latent_dim = latent_dim
		self.shape = shape
		self.encoder = tf.keras.Sequential(
			name="Encoder",
			layers=[
				# tfk.layers.Input(shape=(latent_dim, latent_dim, 1)),
				tfk.layers.Input(shape),
				tfk.layers.Flatten(),
				tfk.layers.Dense(latent_dim, activation='relu'),
			]
		)
		self.decoder = tf.keras.Sequential(
			name="Decoder",
			layers=[
        tfk.layers.Input((latent_dim,)),
        tfk.layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
        tfk.layers.Reshape(shape)
			]
		)


	def call(self, val_x):
		encoded = self.encoder(val_x)
		decoded = self.decoder(encoded)
		return decoded


	def train(self, train_dts, test_dts, epochs, shuffle, callbacks:list=None, multithreading:bool=True):
		"""Wrapper for the fit method."""
		return self.fit(
			train_dts[0], train_dts[1],
			epochs=epochs,
			shuffle=shuffle,
			validation_data=test_dts,
			use_multiprocessing=multithreading,
			callbacks=callbacks,
			batch_size=32,
		)




if __name__ == "__main__":
	shape = x_test.shape[1:]
	latent_dim = 1024
	autoencoder = ReconstructAE(latent_dim, shape)
	autoencoder.compile(
		optimizer=tfk.optimizers.Adam(amsgrad=True, learning_rate=0.001),
    loss=tfk.losses.MeanSquaredError(),
		metrics=['accuracy']
	)
	history = autoencoder.train(
		train_dts=(x_train, x_train),
		test_dts=(x_test, x_test),
		epochs=3,
		shuffle=True,
		callbacks=[autoencoder.cb_checkpoint()]
	)

	# plt.plot(history.history["loss"], label="Training Loss")
	# plt.legend()
	# plt.show()

	# autoencoder.summary()
	autoencoder.save(
		f"{autoencoder.fimodel.filePath()}{autoencoder.name}.keras",
		overwrite=True,
	)

	model = tfk.models.load_model(
		f"{fimlmodels.filePath()}ReconstructAE/ReconstructAE.keras",
		compile=True,
	)
	model.summary()

	encoded_imgs = model.encoder(x_test).numpy()
	decoded_imgs = model.decoder(encoded_imgs).numpy()


	n = 10
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i])
		plt.title("original")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(decoded_imgs[i])
		plt.title("reconstructed")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()