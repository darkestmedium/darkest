# Built-in imports
# import json
import os
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")

# Third-party imports
from api import qtc
from api import np

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk

# Open APi imports
from api import fimlmodels
import api.Darkest as da


(x_train, _), (x_test, _) = tfk.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)





@tfk.saving.register_keras_serializable()
class DenoiseAE(tfk.Model):
	"""Autoencoder for denoising animation data.
	"""
	fimodel = qtc.QFileInfo(f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models")


	# Constructor
	def __init__(self, inputs:int=28, name:str="DenoiseAE", **kwargs):
		super(DenoiseAE, self).__init__(name=name, **kwargs)
		self.weights_path(qtc.QFileInfo(f"/home/{os.getlogin()}/Dropbox/darkest/resources/models/{self.name}/"))
		inputs2 = inputs // 2
		inputs3 = inputs // 4
		self.encoder = tf.keras.Sequential(
			name="Encoder",
			layers=[
				tfk.layers.Input(shape=(inputs, inputs, 1)),
				tfk.layers.Conv2D(inputs2, (3, 3), name="Encoder1", activation='relu', padding='same', strides=2),
				tfk.layers.Conv2D(inputs3, (3, 3), name="Encoder2", activation='relu', padding='same', strides=2)
			]
		)
		self.decoder = tf.keras.Sequential(
			name="Decoder",
			layers=[
				tfk.layers.Conv2DTranspose(inputs3, name="Decoder1", kernel_size=3, activation='relu', padding='same', strides=2),
				tfk.layers.Conv2DTranspose(inputs2, name="Decoder2", kernel_size=3, activation='relu', padding='same', strides=2),
				tfk.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
			]
		)


	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


	def train(self, train_dts, test_dts, epochs, shuffle, callbacks=None, multithreading:bool=True):
		"""Wrapper for the fit method."""
		self.fit(
			train_dts[0], train_dts[1],
			epochs=epochs,
			shuffle=shuffle,
			validation_data=test_dts,
			use_multiprocessing=multithreading,
			callbacks=callbacks,
		)


	def weights_path(self, fimodel):
		DenoiseAE.fimodel = fimodel
		da.iofile.mkdir(fimodel.absoluteFilePath())




if __name__ == "__main__":
	autoencoder = DenoiseAE()
	# EPOCHS = 10
	checkpoint_filepath = f'{autoencoder.fimodel.absoluteFilePath()}checkpoints/'
	da.iofile.mkdir(checkpoint_filepath)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		monitor='accuracy',
		mode='max',
		save_best_only=True,
		verbose=1,
	)

	autoencoder.compile(optimizer='adam', loss=tfk.losses.MeanSquaredError(), metrics=['accuracy'])
	autoencoder.train(
		train_dts=(x_train_noisy, x_train),
		test_dts=(x_test_noisy, x_test),
		epochs=10,
		shuffle=True,
		callbacks=[cp_callback]
	)
	autoencoder.summary()

	autoencoder.save(
		f"{autoencoder.fimodel.filePath()}{autoencoder.name}.keras",
		# save_format="keras",
		overwrite=True,
	)

	model = tfk.models.load_model(
		f"/home/{os.getlogin()}/Dropbox/darkest/resources/ml/models/DenoiseAE/DenoiseAE.keras",
		compile=True,
	)
	model.summary()

	encoded_imgs = model.encoder(x_test_noisy).numpy()
	decoded_imgs = model.decoder(encoded_imgs).numpy()
	n = 10
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original + noise
		ax = plt.subplot(2, n, i + 1)
		plt.title("original + noise")
		plt.imshow(tf.squeeze(x_test_noisy[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		bx = plt.subplot(2, n, i + n + 1)
		plt.title("reconstructed")
		plt.imshow(tf.squeeze(decoded_imgs[i]))
		plt.gray()
		bx.get_xaxis().set_visible(False)
		bx.get_yaxis().set_visible(False)
	plt.show()


