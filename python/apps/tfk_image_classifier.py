# Built-in imports
import os
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")
import random
import argparse
import platform
import math

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Darkest APi imports
from api import np
from api import cv
from api import tf
from api import tfk
from api import qtc

import api.Darkest as da
import api.DarkestMl as daml




# Text formatting
bold = "\033[1m"
end = "\033[0m"

block_plot=False

fimlmodels = qtc.QFileInfo(f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/")




def set_seeds():
  # fix random seeds
  SEED_VALUE = 42

  random.seed(SEED_VALUE)
  np.random.seed(SEED_VALUE)
  tf.random.set_seed(SEED_VALUE)
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

  return

set_seeds()

# Creating a MirroredStrategy for distributed training.
# This strategy effectively replicates the model's layers on each GPU or other available devices,
# syncing their weights after each training step.
DISTRIBUTE_STRATEGY = tf.distribute.MirroredStrategy()
# Printing the number of devices that are in sync with the MirroredStrategy.
# This indicates how many replicas of the model are being trained in parallel.
# print('Number of devices: {}'.format(DISTRIBUTE_STRATEGY.num_replicas_in_sync))




fi_root = qtc.QFileInfo("/home/darkest/Dropbox/code/dataset/classification/opencv-TF-course-project-1-image-classification/dataset")
fi_test, fi_train, fi_valid = da.iofile.listdir(fi_root.absoluteFilePath(), ["*"], filters=qtc.QDir.Dirs, includeSubDirectories=qtc.QDirIterator.NoIteratorFlags)


train_classes = da.iofile.listdir(fi_train.absoluteFilePath(), ["*"], filters=qtc.QDir.Dirs, includeSubDirectories=qtc.QDirIterator.NoIteratorFlags)
valid_classes = da.iofile.listdir(fi_valid.absoluteFilePath(), ["*"], filters=qtc.QDir.Dirs, includeSubDirectories=qtc.QDirIterator.NoIteratorFlags)

train_images = da.iofile.listdir(fi_train.absoluteFilePath())
valid_images = da.iofile.listdir(fi_valid.absoluteFilePath())

# Train Images
fi_train_cow = da.iofile.listdir(train_classes[0].absoluteFilePath())
fi_train_elephant = da.iofile.listdir(train_classes[1].absoluteFilePath())
fi_train_horse = da.iofile.listdir(train_classes[2].absoluteFilePath())
fi_train_spider = da.iofile.listdir(train_classes[3].absoluteFilePath())

# Valid Images
fi_valid_cow = da.iofile.listdir(train_classes[0].absoluteFilePath())
fi_valid_elephant = da.iofile.listdir(train_classes[1].absoluteFilePath())
fi_valid_horse = da.iofile.listdir(train_classes[2].absoluteFilePath())
fi_valid_spider = da.iofile.listdir(train_classes[3].absoluteFilePath())


# print(fi_root.fileName())
# print(f"    {fi_test.fileName()}")
# print(f"    {fi_train.fileName()}")
# [print(f"        {fi.fileName()}") for fi in train_classes]
# print(f"    {fi_valid.fileName()}")
# [print(f"        {fi.fileName()}") for fi in valid_classes]
# print("\n")
# print("Training Classes:")
# [print(f"{fi.fileName()}") for fi in train_classes]
# print("\n")
# print("Validation Classes:")
# [print(f"{fi.fileName()}") for fi in valid_classes]

# print(f"{bold}Number of Training samples: {end}{train_images.__len__()}")
# print(f"{bold}Number of Validation samples: {end}{valid_images.__len__()}")

# print(f"Train cow images: {fi_train_cow.__len__()}")
# print(f"Train elephant images: {fi_train_elephant.__len__()}")
# print(f"Train horse images: {fi_train_horse.__len__()}")
# print(f"Train spider images: {fi_train_spider.__len__()}")


# def show_image(image_path, label):
#   img = cv.imread(image_path)
#   height, width, channels = img.shape

#   plt.imshow(img)
#   plt.title(f"image size: ({width} x {height}, {channels}), target: {label}")
#   plt.axis("off")
#   plt.show()

# target = "cow"
# show_image(fi_train_cow[99].absoluteFilePath(), train_classes[0].fileName())

# target = "elephant"
# show_image(fi_train_elephant[99].absoluteFilePath(), train_classes[1].fileName())

# target = "horse"
# show_image(fi_train_horse[99].absoluteFilePath(), train_classes[2].fileName())

# target = "spider"
# show_image(fi_train_spider[99].absoluteFilePath(), train_classes[3].fileName())






def get_data(fi_train, fi_valid, output, target_size=(224, 224), batch_size=32, data_augmentation=True):

  train_dataset = tfk.utils.image_dataset_from_directory(
    fi_train.absoluteFilePath(), 
    label_mode='categorical',
    color_mode='rgb', 
    batch_size=batch_size, 
    image_size=target_size, 
  )

  valid_dataset = tfk.utils.image_dataset_from_directory(
    fi_valid.absoluteFilePath(), 
    label_mode='categorical',
    color_mode='rgb', 
    batch_size=batch_size,
    image_size=target_size,
  )
  
  if data_augmentation: 
    dts_augmentation = da.iodata.augmentation()
    train_dataset = train_dataset.map(lambda x, y: (dts_augmentation(x), y))
    valid_dataset = valid_dataset.map(lambda x, y: (dts_augmentation(x), y))

  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
  valid_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

  # # Write train dataset to TFRecord
  # train_tfrecord_path = os.path.join(output, "train_dataset.tfrecord")
  # write_tfrecord(train_dataset, train_tfrecord_path)

  # # Write valid dataset to TFRecord
  # valid_tfrecord_path = os.path.join(output, "valid_dataset.tfrecord")
  # write_tfrecord(valid_dataset, valid_tfrecord_path)

  return train_dataset, valid_dataset
  # return train_tfrecord_path, valid_tfrecord_path





@dataclass
class TrainingConfig:
  # Defining the batch size for model training.
  # The batch size is set to be some integer times the  number of devices in synchronization as per the distributed strategy.
  # This means that the overall batch of data is divided equally across all the devices used in the distributed training.
  # By scaling the batch size with the number of replicas (devices), each device processes a batch of size, in this case, 4.
  
  # This approach helps in efficient utilization of the computational power of all the devices involved in training.
  BATCH_SIZE: int = 4 * DISTRIBUTE_STRATEGY.num_replicas_in_sync

  # For tensorboard logging and saving checkpoints
  root_log_dir = os.path.join("Logs_Checkpoints", "Model_logs")
  root_checkpoint_dir = os.path.join("Logs_Checkpoints", "Model_checkpoints")

  # Current log and checkpoint directory.
  log_dir = "version_0"
  checkpoint_dir = "version_0"




@dataclass
class DatasetConfig:
  DATA_ROOT: str = fi_root.absoluteFilePath()
  DATA_SHAPE: tuple = (128, 256, 3)
  NUM_CLASSES: int = 4




def setup_log_directory(training_config=TrainingConfig()):
  """Tensorboard Log and Model checkpoint directory Setup"""
  if os.path.isdir(training_config.root_log_dir):
    # Get all folders numbers in the root_log_dir
    folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(training_config.root_log_dir)]
    # Find the latest version number present in the log_dir
    last_version_number = max(folder_numbers)
    # New version name
    version_name = f"version_{last_version_number + 1}"
  else:
    version_name = training_config.log_dir

  # Update the training config default directory 
  training_config.log_dir = os.path.join(training_config.root_log_dir, version_name)
  training_config.checkpoint_dir = os.path.join(training_config.root_checkpoint_dir, version_name)

  # Create new directory for saving new experiment version
  os.makedirs(training_config.log_dir, exist_ok=True)
  os.makedirs(training_config.checkpoint_dir, exist_ok=True)

  print(f"Logging at: {training_config.log_dir}")
  print(f"Model Checkpoint at: {training_config.checkpoint_dir}")
  
  return training_config, version_name




def plot_history(
  train_loss=None,
  val_loss=None,
  train_metric=None,
  val_metric=None,
  colors=["blue", "green"],
  loss_legend_loc="upper center",
  acc_legend_loc="upper left",
  fig_size=(15, 10),
):

  plt.rcParams["figure.figsize"] = fig_size
  fig = plt.figure()
  fig.set_facecolor("white")

  # Loss Plots
  plt.subplot(2, 1, 1)

  train_loss_range = range(len(train_loss))
  plt.plot(
    train_loss_range,
    train_loss,
    color=f"tab:{colors[0]}",
    label=f"Train Loss",
  )

  valid_loss_range = range(len(val_loss))
  plt.plot(
    valid_loss_range,
    val_loss,
    color=f"tab:{colors[1]}",
    label=f"Valid Loss",
  )

  plt.ylabel("Loss")
  plt.legend(loc=loss_legend_loc)
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.grid(True)
  plt.title("Training and Validation Loss")

  # Accuracy Plots
  plt.subplot(2, 1, 2)

  train_metric_range = range(len(train_metric))
  plt.plot(
    train_metric_range,
    train_metric,
    color=f"tab:{colors[0]}",
    label=f"Train Accuracy",
  )

  val_metric_range = range(len(val_metric))
  plt.plot(
    val_metric_range,
    val_metric,
    color=f"tab:{colors[1]}",
    label=f"Valid Accuracy",
  )

  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.legend(loc=acc_legend_loc)
  plt.grid(True)
  plt.title("Training and Validation Accuracy")

  plt.show(block=block_plot)

  return




@tfk.saving.register_keras_serializable()
class Classifier4(daml.dnnmodel):
  """Autoencoder for denoising animation data.
  """
  fimodel = ...
  fimodelcp = ...

  # Constructor 
  def __init__(self, models_path:str, num_classes=4, input_shape=(224, 224, 3), name:str="Classifier4", **kwargs):
    super(Classifier4, self).__init__(name=name, **kwargs)

    Classifier4.fimodel = qtc.QFileInfo(f"{models_path}{name}/")
    Classifier4.fimodelcp = qtc.QFileInfo(f"{Classifier4.fimodel.absoluteFilePath()}checkpoint/")
    [da.iofile.mkdir(fi.absoluteFilePath()) for fi in [Classifier4.fimodel, Classifier4.fimodelcp]]  # Create paths -.-

    self.sequential = tfk.Sequential(
      name="Sequential",
      layers=[
        tfk.layers.Rescaling(1./255, input_shape=input_shape),

        tfk.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tfk.layers.MaxPooling2D(),

        tfk.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tfk.layers.MaxPooling2D(),

        tfk.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tfk.layers.MaxPooling2D(),

        tfk.layers.Flatten(),

        tfk.layers.Dense(128, activation="relu"),
        tfk.layers.Dense(num_classes, activation="softmax")
      ]
    )


  def call(self, inputs):
    """Custom call method."""
    return self.sequential(inputs)


  def train(self, train_dts, validation_dts, callbacks=None, epochs:int=10, batch_size=32, multithreading:bool=True, workers:int=1):
    """Wrapper for the fit method.
    """
    return self.fit(
      train_dts,
      validation_data=validation_dts,
      epochs=epochs,
      callbacks=callbacks,
      workers=workers,
      use_multiprocessing=multithreading,
      batch_size=batch_size,
    )




def get_sample_predictions(*, model, dataset, total=15):

  imgs = []
  ground_truths = []
  probs = []
  predictions = []

  idx_to_cls = {0: "cow", 1: "elephant", 2: "horse", 3: "spider"}

  print("Generating Predictions...")
  for data, target in dataset:
    model_predictions = model.predict_on_batch(data)
    cls_predicted = np.argmax(model_predictions, axis=-1)
    cls_probs = np.max(model_predictions, axis=-1)

    imgs.extend(data.numpy() / 255.)
    ground_truths.extend(target.numpy())
    predictions.extend(cls_predicted)
    probs.extend(cls_probs)
    
    # Displaying only 15 images 
    if data.shape[0] >= total: 
      break

  plt.style.use("default")
  plt.rcParams["figure.figsize"] = (18, 9)
  fig = plt.figure()
  fig.set_facecolor("white")

  for idx in range(total):
    plt.subplot(3, 5, idx + 1)
    img = imgs[idx]
    plt.imshow(img)

    plt.title(f"P:{idx_to_cls[predictions[idx]]}({probs[idx]:.2}), T:{idx_to_cls[ground_truths[idx]]}")
    plt.axis("off")

  fig.savefig("sample_predictions.png")
  plt.show(block=block_plot)
  
  del imgs, ground_truths, probs, predictions
  return




def lr_schedule(epoch, initial_lr=0.01):
  drop = 0.5
  epochs_drop = 10
  return initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))



def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/", help="Path to the model file and checkpoint.")
  parser.add_argument("-dfp", "--datasetFilePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/ml/models/", help="Path to the model file and checkpoint.")
  parser.add_argument("-ts", "--targetSize", type=tuple, default=(128, 256, 3), help="Target size.")
  parser.add_argument("-ep", "--epochs", type=int, default=1, help="Number of epochs to train for.")
  parser.add_argument("-lr", "--learningRate", type=float, default=0.01, help="Learning rate.")
  parser.add_argument("-bs", "--batchSize", type=int, default=4*DISTRIBUTE_STRATEGY.num_replicas_in_sync, help="Batch size.")
  parser.add_argument("-mp", "--multiProcessing", type=bool, default=True if platform.system()=="Linux" else False, help="Wheter or not to use multi threading.")
  parser.add_argument("-now", "--numberOfWorkers", type=int, default=4, help="Number of workers to use for training.")
  parser.add_argument("-noc", "--numberOfClasses", type=int, default=4, help="Number of classes to use for training / classification.")
  parser.add_argument("-da", "--dataAugmentation", type=bool, default=True, help="Wheter or not to use data augmentation.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()

  # Get training and validation datasets
  train_dataset, valid_dataset = get_data(
    fi_train, fi_valid, fi_root.absoluteFilePath(),
    target_size=DatasetConfig.DATA_SHAPE[:2],
    batch_size=args.batchSize,
    data_augmentation=args.dataAugmentation,
  )


  # Start a context manager using the distributed strategy previously defined.
  # This scope ensures that the operations defined within it are distributed across the available devices as per the strategy.
  with DISTRIBUTE_STRATEGY.scope():
    model = Classifier4(args.filePath, num_classes=args.numberOfClasses, input_shape=args.targetSize)
    model.compile(
      optimizer=tfk.optimizers.Adam(learning_rate=args.learningRate),
      loss=tfk.losses.CategoricalCrossentropy(),
      # loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )


  history = model.train(
    train_dts=train_dataset,
    validation_dts=valid_dataset,
    epochs=args.epochs,
    callbacks=[
      model.cb_checkpoint(),
      # tfk.callbacks.LearningRateScheduler(lr_schedule)
    ],
    multithreading=args.multiProcessing,
    batch_size=args.batchSize,
    workers=args.numberOfWorkers,
  )

  model.save(f"{model.fimodel.filePath()}{model.name}.keras", overwrite=True)
  tfk.utils.plot_model(model.sequential, to_file="model.png", show_shapes=True)

  # Reload
  model_reloaded = tfk.models.load_model(f"{model.fimodel.filePath()}/{model.name}.keras", compile=True)
  model_reloaded.sequential.summary()


  valid_dataset = tfk.utils.image_dataset_from_directory(
    fi_valid.absoluteFilePath(), 
    label_mode='int',
    color_mode='rgb', 
    batch_size=args.batchSize, 
    image_size=DatasetConfig.DATA_SHAPE[:2], 
    shuffle=True, # shuffling to show images from all classes
  )
  get_sample_predictions(model=model_reloaded, dataset=valid_dataset)

