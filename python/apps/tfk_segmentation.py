# Built-in imports
import os
import sys; sys.path.append(f"/home/{os.getlogin()}/Dropbox/code/darkest/python")  # import api 'cause hate venvs
import argparse
import platform
import math
import glob as glob

# Third-party imports
import pandas as pd
import zipfile 
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

# from tensorflow.keras.utils import Sequence
from dataclasses import dataclass
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# Darkest APi imports
from api import np
from api import cv
from api import tf
from api import tfk
from api import qtc

import api.Darkest as da
import api.DarkestMl as daml



DISTRIBUTE_STRATEGY = tf.distribute.MirroredStrategy()


@dataclass(frozen=True)
class DatasetConfig:
  NUM_CLASSES: int = 10
  IMG_WIDTH:   int = 256
  IMG_HEIGHT:  int = 256

  DATA_TRAIN_IMAGES: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/train/images/*.jpg"
  DATA_TRAIN_LABELS: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/train/masks/*.png"
  DATA_VALID_IMAGES: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/valid/images/*.jpg"
  DATA_VALID_LABELS: str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/valid/masks/*.png"
  DATA_TEST_IMAGES:  str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/test/images/*.jpg"
  DATA_TEST_LABELS:  str = "/home/darkest/Dropbox/code/resources/dataset/segmentation/FloodNet-Supervised-540p_v1.0/test/masks/*.png"

  id2color = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # Building Flooded
    2: (200, 90, 90),  # Non-Flooded Building
    3: (128, 128, 0),  # Road Flooded 
    4: (155, 155, 155),  # Non-Flooded Road
    5: (0, 255, 255),  # Water
    6: (55, 0, 255),  # Tree
    7: (255, 0, 255),  # Vehicle
    8: (245, 245, 0),  # Pool
    9: (0, 255, 0),  # Grass
  }








# Function to convert a single channel mask representation to an RGB mask.
def num_to_rgb(num_arr, color_map=DatasetConfig.id2color):
  single_layer = np.squeeze(num_arr)
  output = np.zeros(num_arr.shape[:2]+(3,))
  for k in color_map.keys():
    output[single_layer==k] = color_map[k]

  return np.float32(output) / 255. # return a floating point array in range [0.0, 1.0]



# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
  alpha = 1.0 # Transparency for the original image.
  beta  = 0.7 # Transparency for the segmentation map.
  gamma = 0.0 # Scalar added to each sum.
  
  segmented_image = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  image = cv.addWeighted(image, alpha, segmented_image, beta, gamma, image)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  
  return np.clip(image, 0.0, 1.0)





def display_image_and_mask(data_list, color_mask=False, color_map=DatasetConfig.id2color):
  plt.figure(figsize=(16, 6))
  title = ['GT Image', 'GT Mask', 'Overlayed Mask']

  grayscale_gt_mask = data_list[1]

  # Create RGB segmentation map from grayscale segmentation map.
  rgb_gt_mask = num_to_rgb(data_list[1], color_map=color_map)

  # Create the overlayed image.
  overlayed_image = image_overlay(data_list[0], rgb_gt_mask)
  
  data_list.append(overlayed_image)
  
  for i in range(len(data_list)):
    plt.subplot(1, len(data_list), i+1)
    plt.title(title[i])
    if title[i] == 'GT Mask':
      if color_mask:
        plt.imshow(np.array(rgb_gt_mask))
      else:
        plt.imshow(np.array(grayscale_gt_mask))
    else:
      plt.imshow(np.array(data_list[i]))
    plt.axis('off')

  plt.show()




# def create_datasets(aug=False, split_ratio:float=0.8, batch_size:int=8):
#   """Creates the train and validation datasets from given images - stripped kaggle ds.
#   """
#   data_images = sorted(glob.glob(f"{DatasetConfig.DATA_TRAIN_IMAGES}"))
#   data_masks  = sorted(glob.glob(f"{DatasetConfig.DATA_TRAIN_LABELS}"))
#   split_index = int(data_images.__len__() * split_ratio)
#   return (
#     da.iods_segmentation(  # Train data loader.
#       batch_size=batch_size,
#       image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
#       image_paths=data_images[:split_index],
#       mask_paths=data_masks[:split_index],
#       num_classes=DatasetConfig.NUM_CLASSES,
#       color_map=DatasetConfig.id2color,
#       apply_aug=aug,
#     ),
#     da.iods_segmentation(  # Validation data loader.
#       batch_size=batch_size,
#       image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
#       image_paths=data_images[split_index:],
#       mask_paths=data_masks[split_index:],
#       num_classes=DatasetConfig.NUM_CLASSES,
#       color_map=DatasetConfig.id2color,
#       apply_aug=False,
#     )
#   )
def create_datasets(aug=False, split_ratio:float=0.8, batch_size:int=8):
  """Creates the train and validation datasets from given images - full ds version.
  """
  train_images = sorted(glob.glob(DatasetConfig.DATA_TRAIN_IMAGES))
  train_masks  = sorted(glob.glob(DatasetConfig.DATA_TRAIN_LABELS))
  valid_images = sorted(glob.glob(DatasetConfig.DATA_VALID_IMAGES))
  valid_masks  = sorted(glob.glob(DatasetConfig.DATA_VALID_LABELS))

  return (
    da.iods_segmentation(  # Train data loader.
      batch_size=batch_size,
      image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
      image_paths=train_images,
      mask_paths=train_masks,
      num_classes=DatasetConfig.NUM_CLASSES,
      color_map=DatasetConfig.id2color,
      apply_aug=aug,
    ),
    da.iods_segmentation(  # Validation data loader.
      batch_size=batch_size,
      image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
      image_paths=valid_images,
      mask_paths=valid_masks,
      num_classes=DatasetConfig.NUM_CLASSES,
      color_map=DatasetConfig.id2color,
      apply_aug=False,
    )
  )



def dice_coefficient(y_true, y_pred, smooth=1e-7):
  intersection = tf.reduce_sum(y_true * y_pred)
  union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
  dice = (2.0 * intersection + smooth) / (union + smooth)
  return dice


def dice_loss(y_true, y_pred):
  return 1.0 - dice_coefficient(y_true, y_pred)


def dice_coefficient_metric(y_true, y_pred):
  return dice_coefficient(y_true, y_pred)


def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
  x = tfk.layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias)(block_input)
  x = tfk.layers.BatchNormalization()(x)
  return tfk.layers.Activation('relu')(x)




def DilatedSpatialPyramidPooling(dspp_input):
  dims = dspp_input.shape
  # Create a 1x1 feature map using AveragePooling2D.
  x = tfk.layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
  x = convolution_block(x, kernel_size=1, use_bias=True)

  # Upsample the feature map to the original size.
  out_pool = tfk.layers.UpSampling2D(size=(dims[1], dims[2]), interpolation="bilinear")(x)

  # Create feature maps of the same shape with different dilation rates.
  out_1  = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
  out_6  = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
  out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
  out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

  # Combine all the feature maps and process them through a 1x1 convolutional block.
  x = tfk.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
  output = convolution_block(x, kernel_size=1)
  
  return output




def deeplabv3plus(num_classes, shape):

  model_input = tfk.layers.Input(shape=shape)
  preprocessing = tfk.applications.resnet50.preprocess_input(model_input)
  backbone = tfk.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_tensor=preprocessing)

  # Set all layers in the backbone as trainable.
  for layer in backbone.layers:
    layer.trainable = True

  # Obtain a (lower resolution) feature map from the backbone.
  # Shape: (14, 14, 256)
  input_a = backbone.get_layer("conv4_block6_2_relu").output

  # Pass through Atrous Spatial Pyramid Pooling to obtain features at various scales.
  input_a = DilatedSpatialPyramidPooling(input_a)

  # Upsample the concatenated features.
  input_a = tfk.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(input_a)

  # Obtain a second (higher resolution) feature map from the backbone and apply convolution.
  # Shape: (56, 56, 64)
  input_b = backbone.get_layer("conv2_block3_2_relu").output
  input_b = convolution_block(input_b, num_filters=256, kernel_size=1)

  # Concatenate both sets of feature maps and perform final decoder processing.
  x = tfk.layers.Concatenate(axis=-1)([input_a, input_b])
  x = convolution_block(x)
  x = convolution_block(x)
  x = tfk.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

  # Apply 1x1 convolution to limit the depth of the feature maps to the number of classes.
  outputs = tfk.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)

  model_output = tfk.layers.Activation('softmax')(outputs)
  model = tfk.Model(inputs=model_input, outputs=model_output)

  return model




# Loss functions:

# def cross_entropy_loss(y_true, y_pred):
#   return tf.reduce_mean(tfk.losses.categorical_crossentropy(y_true, y_pred))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
  epsilon = 1e-8
  y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
  focal_loss = - (alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred))
  return tf.reduce_mean(focal_loss)

def iou_loss(y_true, y_pred):
  intersection = tf.reduce_sum(y_true * y_pred)
  union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
  iou = intersection / union
  return 1 - iou

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
  return 1 - (numerator + 1) / (denominator + 1)





def mean_iou(y_true, y_pred):
  """
  Arguments:
  y_true (ndarray or tensor): Ground truth mask (G). Shape: (batch_size, height, width) Sparse representation of segmentation mask.
  y_pred (ndarray or tensor): Prediction (P) from the model with or without softmax. Shape: (batch_size, height, width, num_classes).

  return (scalar): Classwise mean IoU Metric.
  """
  
  # Get total number of classes from model output.
  num_classes = y_pred.shape[-1]

  # Convert single channel (sparse) ground truth labels to one-hot encoding for metric computation.
  y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, axis=-1)

  # Convert multi-channel predicted output to one-hot encoded thresholded output for metric computation. 
  y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes, axis=-1)

  # Axes corresponding to image width and height: [B, H, W, C].
  axes = (1, 2)

  # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
  intersection = tf.math.reduce_sum(y_true * y_pred, axis=axes)

  # Total Sum: |G| + |P|. Shape: (batch_size, num_classes)
  total = tf.math.reduce_sum(y_true, axis=axes) + tf.math.reduce_sum(y_pred, axis=axes)

  # Union: Shape: (batch_size, num_classes)
  union = total - intersection

  # Boolean (then converted to float) value for each class if it is present or not. 
  # Shape: (batch_size, num_classes)
  is_class_present =  tf.cast(tf.math.not_equal(total, 0), dtype=tf.float32)

  # Sum along axis(1) to get number of classes in each image.
  # Shape: (batch_size,)
  num_classes_present = tf.math.reduce_sum(is_class_present, axis=1)

  # Here, we use tf.math.divide_no_nan() to prevent division by 0 (i.e., 0/0 = 0).
  # Shape: (batch_size, num_classes)
  iou = tf.math.divide_no_nan(intersection, union)

  # IoU per image. Average over the total number of classes present in y_true and y_pred.
  # Shape: (batch_size,)
  iou = tf.math.reduce_sum(iou, axis=1) / num_classes_present
  
  # Compute the mean across the batch axis. Shape: Scalar
  mean_iou = tf.math.reduce_mean(iou)
  
  return mean_iou


def rle_encode(mask):
  """Encodes a binary mask into its Run-Length Encoding (RLE) format.
  :param mask: A binary mask image (numpy array).
  """
  mask = mask.astype('int32')
  pixels = mask.reshape(-1, order='C')
  rle = []
  prev = -1
  cnt = 0
  for pix in pixels:
    if pix != prev:
      if cnt > 0:
        rle.append((prev, cnt))
      cnt = 1
      prev = pix
    elif cnt > 0:
      cnt += 1
    else:
      pass
  if cnt > 0:
    rle.append((prev, cnt))
  return rle


def decode_rle(rle_str):
  """
  Decodes a Run-Length Encoding (RLE) encoded string into a binary mask.
  :param rle_str: A RLE encoded string.
  """
  mask = np.zeros((256, 256), dtype=np.uint8)
  cnt, pos = 0, 0
  for t in rle_str.split(' '):
    cnt, _ = map(int, t.split(','))
    mask[pos:pos + cnt] = 1
    pos += cnt
  return mask


def get_submission(predictions, save_path):
  """Generates a submission CSV file.
  :param predictions: A 4D numpy array containing the model's predictions (batch_size, num_classes, height, width).
  :param save_path: The path to save the submission CSV file.
  """
  df = pd.DataFrame()
  img_ids = []
  masks = []
  for img_id, pred in enumerate(predictions):
    for class_id in range(10):
      mask = pred[class_id].squeeze()
      mask = cv.resize(mask, (256, 256), interpolation=cv.INTER_NEAREST)
      encoded_mask = rle_encode(mask)
      img_id_str = f'{img_id}_{class_id}'
      img_ids.append(img_id_str)
      masks.append(' '.join(map(str, encoded_mask)))
  df['IMG_ID'] = img_ids
  df['EncodedString'] = masks
  df.to_csv(save_path, index=False)




def plot_results(metrics, ylabel, ylim, epochs, metric_name=None, color=None, block_plot=False):
  fig, ax = plt.subplots(figsize=(18, 5))

  if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
    metrics = [metrics,]
    metric_name = [metric_name,]
      
  for idx, metric in enumerate(metrics):
    ax.plot(metric, color=color[idx])
  
  plt.xlabel("Epoch")
  plt.ylabel(ylabel)
  plt.title(ylabel)
  plt.xlim([0, epochs-1])
  plt.ylim(ylim)
  # Tailor x-axis tick marks
  ax.xaxis.set_major_locator(MultipleLocator(5))
  ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
  ax.xaxis.set_minor_locator(MultipleLocator(1))
  plt.grid(True)
  plt.legend(metric_name)   
  plt.show(block=block_plot)
  plt.close()


def inference(model, dataset, num_batches=3):
  num_batches_to_process = num_batches
  predictions = []

  for idx, data in enumerate(dataset):
    batch_img, batch_mask = data[0], data[1]
    pred_all = (model.predict(batch_img)).astype('float32')
    pred_all = pred_all.argmax(-1)
    batch_img = (batch_img).astype('uint8')

    # predictions.append(pred_all)

    # Save the predictions to a NumPy array
    if idx == num_batches_to_process:
      pred_array = np.stack(pred_all)
      predictions.append(pred_array)
      np.save('predictions.npy', pred_array)
      break

    # # Display the predictions using matplotlib
    # for i in range(0, len(batch_img)):
    #   fig = plt.figure(figsize=(20,8))

    #   # Display the original image.
    #   ax1 = fig.add_subplot(1,4,1)
    #   ax1.imshow(batch_img[i])
    #   ax1.title.set_text('Actual frame')
    #   plt.axis('off')
      
    #   # Display the ground truth mask.
    #   true_mask = batch_mask[i]
    #   ax2 = fig.add_subplot(1,4,2)
    #   ax2.set_title('Ground truth labels')
    #   ax2.imshow(true_mask)
    #   plt.axis('off')
      
    #   # Display the predicted segmentation mask. 
    #   pred_mask = pred_all[i]
    #   ax3 = fig.add_subplot(1,4,3)
    #   ax3.set_title('Predicted labels')
    #   ax3.imshow(pred_mask)
    #   plt.axis('off')

    #   # Display the predicted segmentation mask overlayed on the original image.
    #   pred_mask_rgb = num_to_rgb(pred_all[i], color_map=DatasetConfig.id2color)
    #   overlayed_image = image_overlay((batch_img[i]), np.array(pred_mask_rgb.astype('uint8')))
    #   ax4 = fig.add_subplot(1,4,4)
    #   ax4.set_title('Overlayed image')
    #   ax4.imshow(overlayed_image)
    #   plt.axis('off')
      
    #   plt.show()

  np.save('predictions_all.npy', predictions)
  return predictions




def syntax_creator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", type=bool, default=True, help="Wheter or not to perform training.")
  parser.add_argument("-fp", "--filePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/darkest/resources/models/", help="Path to the model file and checkpoint.")
  parser.add_argument("-dfp", "--datasetFilePath", type=str, default=f"/home/{os.getlogin()}/Dropbox/code/resources/dataset/segmentation/opencv-tensorflow-course-project-3-segmentation/Project_3_FloodNet_Dataset/", help="Path to the dataset.")
  parser.add_argument("-ts", "--targetSize", type=tuple, default=(256, 256, 3), help="Target size.")
  parser.add_argument("-ep", "--epochs", type=int, default=5, help="Number of epochs to train for.")
  parser.add_argument("-lr", "--learningRate", type=float, default=0.01, help="Learning rate.")
  parser.add_argument("-bs", "--batchSize", type=int, default=4*DISTRIBUTE_STRATEGY.num_replicas_in_sync, help="Batch size.")
  parser.add_argument("-noc", "--numberOfClasses", type=int, default=10, help="Number of classes to use for training / classification.")
  parser.add_argument("-da", "--dataAugmentation", type=bool, default=True, help="Wheter or not to use data augmentation.")
  parser.add_argument("-mp", "--multiProcessing", type=bool, default=True if platform.system()=="Linux" else False, help="Wheter or not to use multi threading.")
  parser.add_argument("-now", "--numberOfWorkers", type=int, default=4, help="Number of workers to use for training.")
  return parser.parse_args()




if __name__ == "__main__":
  args = syntax_creator()

  train_ds, valid_ds = create_datasets(aug=True)

  # Visualize dataset
  # for i, (images, masks) in enumerate(train_ds):
  #   if i == 3: break
  #   image, mask = images[0], masks[0]
  #   display_image_and_mask([image, mask], color_mask=False, color_map=DatasetConfig.id2color)

  model = deeplabv3plus(num_classes=args.numberOfClasses, shape=args.targetSize)

  model.compile(  
    optimizer=tfk.optimizers.Adam(learning_rate=args.learningRate),   
    loss=tfk.losses.SparseCategoricalCrossentropy(),
    # loss=focal_loss,
    # loss=tfk.losses.binary_crossentropy,
    # loss=tfk.losses.CategoricalCrossentropy(),
    # loss=iou_loss,
    # loss=dice_loss,
    metrics=['accuracy', mean_iou],
  )

  # model.summary()
  history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=args.epochs, 
    workers=args.numberOfWorkers,
    use_multiprocessing=args.multiProcessing,
    callbacks=[
      daml.dnnmodel.cb_checkpoint(f"{args.filePath}DeepLabV3/checkpoint.h5"),
      tfk.callbacks.EarlyStopping(monitor="loss", patience=5),
    ],
    verbose=1,
  )

  # Visualize training
  # Pixel accuracy.
  train_acc = history.history["accuracy"]
  valid_acc = history.history["val_accuracy"]
  # Mean IoU.
  train_iou = history.history["mean_iou"]
  valid_iou = history.history["val_mean_iou"]
  plot_results(
    [train_acc, valid_acc],
    ylabel="Pixel Accuracy",
    ylim = [0.5, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
    epochs=args.epochs,
  )
  plot_results(
    [train_iou, valid_iou],
    ylabel="Mean IoU",
    ylim = [0.0, 1.0],
    metric_name=["Training IoU", "Validation IoU"],
    color=["g", "b"],
    epochs=args.epochs,
  )


  # if args.train:
  #   # model = tfk.models.load_model(
  #   #   f"{args.filePath}DeepLabV3/checkpoint.h5",
  #   #   custom_objects={'mean_iou':mean_iou},
  #   #   compile=True
  #   # )

  #   history = model.fit(
  #     train_ds,
  #     validation_data=valid_ds,
  #     epochs=args.epochs, 
  #     workers=args.numberOfWorkers,
  #     use_multiprocessing=args.multiProcesmercurysing,
  #     callbacks=[
  #       daml.dnnmodel.cb_checkpoint(f"{args.filePath}DeepLabV3/checkpoint.h5"),
  #       tfk.callbacks.EarlyStopping(monitor="loss", patience=5),
  #     ],
  #     verbose=1,
  #   )

  #   # Visualize training
  #   # Pixel accuracy.
  #   train_acc = history.history["accuracy"]
  #   valid_acc = history.history["val_accuracy"]
  #   # Mean IoU.
  #   train_iou = history.history["mean_iou"]
  #   valid_iou = history.history["val_mean_iou"]
  #   plot_results(
  #     [train_acc, valid_acc],
  #     ylabel="Pixel Accuracy",
  #     ylim = [0.5, 1.0],
  #     metric_name=["Training Accuracy", "Validation Accuracy"],
  #     color=["g", "b"],
  #     epochs=args.epochs,
  #   )
  #   plot_results(
  #     [train_iou, valid_iou],
  #     ylabel="Mean IoU",
  #     ylim = [0.0, 1.0],
  #     metric_name=["Training IoU", "Validation IoU"],
  #     color=["g", "b"],
  #     epochs=args.epochs,
  #   )


  # trained_model = tfk.models.load_model(
  #   f"{args.filePath}DeepLabV3/checkpoint.h5",
  #   custom_objects={'mean_iou':mean_iou},
  #   compile=True
  # )

  # # trained_model.summary()

  # evaluate = trained_model.evaluate(valid_ds)
  # print(f"Model evaluation accuracy: {evaluate[1]*100.:.3f}")
  # print(f"Model evaluation mean IoU: {evaluate[2]*100.:.3f}")

  # # Inference
  # inference(trained_model, valid_ds)

  # # Load the saved model predictions Generate submission CSV file
  # predictions = np.load('predictions_all.npy')
  # save_path = 'submission.csv'
  # get_submission(predictions, save_path)
  # print(f"Submission CSV file saved to '{save_path}'.")



