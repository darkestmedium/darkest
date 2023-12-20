# Built-in imports
import os
import ssl
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")

# Third-party imports
from api import qtc
from api import np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tfk

# Darkest APi imports
import api.Darkest as da


# Global paths

fimlmodels = qtc.QFileInfo(f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/")


# Reference https://keras.io/api/datasets/fashion_mnist/
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0




@tfk.saving.register_keras_serializable()
class DNN(tfk.Model):
  """Keras Model subclass wrapper with convinience methods.
  """
  fimodel = ...
  fimodelcp = ...

  @classmethod
  def cb_checkpoint(cls, path:str=None, best:bool=True):
    """Call back for saving the best model during training.
    """
    if not path: path = cls.fimodelcp.absoluteFilePath()
    da.iofile.mkdir(path)
    return tfk.callbacks.ModelCheckpoint(
      filepath=path,
      monitor="accuracy",
      mode="max",
      save_best_only=best,
      verbose=1,
    )




@tfk.saving.register_keras_serializable()
class MLP(DNN):
  """Autoencoder for denoising animation data.
  """

  # Constructor 
  def __init__(self, inputs:int=28, name:str="MLP", **kwargs):
    super(MLP, self).__init__(name=name, **kwargs)
    MLP.fimodel = qtc.QFileInfo(f"{fimlmodels.absoluteFilePath()}{name}/")
    MLP.fimodelcp = qtc.QFileInfo(f"{MLP.fimodel.absoluteFilePath()}checkpoint/")
    [da.iofile.mkdir(fi.absoluteFilePath()) for fi in [MLP.fimodel, MLP.fimodelcp]]  # Create paths -.-

    self.model = tfk.Sequential(
      name="Model",
      layers=[
        tfk.layers.Flatten(input_shape=(inputs, inputs)),
        tfk.layers.Dense(128, name="Dense128", activation='relu'),
        tfk.layers.Dense(10, name="Dense10")
      ]
    )


  def call(self, inputs):
    """Custom call method."""
    return self.model(inputs)


  def train(self, train_dts, test_dts, callbacks=None, epochs:int=10, batch_size=32, multithreading:bool=True):
    """Wrapper for the fit method."""
    return self.fit(
      train_dts[0], train_dts[1],
      epochs=epochs,
      validation_data=test_dts,
      use_multiprocessing=multithreading,
      callbacks=callbacks,
      batch_size=batch_size,
    )




def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel(
    "{} {:2.0f}% ({})".format(class_names[predicted_label],
    100*np.max(predictions_array),
    class_names[true_label]),
    color=color
  )


def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')




if __name__ == "__main__":

  mlp = MLP()
  mlp.compile(
    optimizer='adam',
    loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  history = mlp.train(
    train_dts=(train_images, train_labels),
    test_dts=(test_images, test_labels),
    epochs=3,
    callbacks=mlp.cb_checkpoint(),
  )
  mlp.save(
    f"{mlp.fimodel.filePath()}{mlp.name}.keras",
    overwrite=True
  )
  mlp.summary()

  model = tfk.models.load_model(f"{mlp.fimodel.filePath()}/{mlp.name}.keras", compile=True)
  model.summary()

  # Plot Loss
  plt.figure(figsize=[15,5])
  plt.plot(history.history['loss'], 'g')
  plt.plot(history.history['val_loss'], 'b')

  plt.xlabel('Epochs')
  plt.ylabel('Loss')

  plt.legend(['Training', 'Validation'], loc='upper right')
  plt.grid(True)

  # Plot Accuracy
  plt.figure(figsize=[15,5])
  plt.plot(history.history['accuracy'], 'g')
  plt.plot(history.history['val_accuracy'], 'b')

  plt.ylim([0.5, 1])

  plt.xlabel('Epochs')
  plt.ylabel('Acc')

  plt.legend(['Training', 'Validation'], loc='lower right')
  plt.grid(True)

  plt.show()


  probability_model = tf.keras.Sequential([model, tfk.layers.Softmax()])
  predictions = probability_model.predict(test_images)
  predictions[0]
  np.argmax(predictions[0])


  # Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
  plt.tight_layout()
  plt.show()



