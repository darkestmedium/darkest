import cv2
import os
import shutil
import numpy as np




def resize_dataset(input_folder, output_folder, target_width, target_height):
  # Create the output folder if it doesn't exist
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Traverse the input folder and its subfolders
  for root, dirs, files in os.walk(input_folder):
    # Construct the corresponding output folder structure
    relative_path = os.path.relpath(root, input_folder)
    output_path = os.path.join(output_folder, relative_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    # Process each file in the current folder
    for file_name in files:
      # Construct the full path to the input file
      input_path = os.path.join(root, file_name)

      # Read the image using OpenCV
      image = cv2.imread(input_path)

      if image is not None:
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Construct the full path to the output file
        output_file_path = os.path.join(output_path, file_name)

        # Save the resized image
        cv2.imwrite(output_file_path, resized_image)



def recolor_masks_in_folder(input_folder, output_folder):
  """
  id2greyscale = {
    0: (0, 0, 0),        # Background
    1: (1, 1, 1),        # Building Flooded
    2: (2, 2, 2),        # Non-Flooded Building
    3: (3, 3, 3),        # Road Flooded 
    4: (4, 4, 4),        # Non-Flooded Road
    5: (5, 5, 5),        # Water
    6: (6, 6, 6),        # Tree
    7: (7, 7, 7),        # Vehicle
    8: (8, 8, 8),        # Pool
    9: (9, 9, 9),        # Grass
  }

  """

  id2color = {
    0: (0, 0, 0),        # Background
    1: (0, 0, 255),      # Building Flooded
    2: (90, 90, 200),    # Non-Flooded Building
    3: (0, 128, 128),    # Road Flooded 
    4: (155, 155, 155),  # Non-Flooded Road
    5: (255, 255, 0),    # Water
    6: (255, 0, 55),     # Tree
    7: (255, 0, 255),    # Vehicle
    8: (0, 245, 245),    # Pool
    9: (0, 255, 0),      # Grass
  }

  # Ensure the output folder exists
  os.makedirs(output_folder, exist_ok=True)

  # Loop through each file in the input folder
  for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
      # Construct input and output paths
      input_path = os.path.join(input_folder, filename)
      output_path = os.path.join(output_folder, filename)
      mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

      # Create an RGB image using the color mapping
      colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
      for idx, color in id2color.items():
        colored_mask[np.where(mask == idx)] = color

      # Save the recolored image using OpenCV
      cv2.imwrite(output_path, colored_mask)



if __name__ == "__main__":
  # Set the paths and parameters
  input_folder = "/home/darkest/Downloads/FloodNet-Supervised_v1.0"
  output_folder = "/home/darkest/Downloads/FloodNet-Supervised-540p_v1.0"
  target_width = 720
  target_height = 540

  # 720 x 540

  # Resize the dataset
  resize_dataset(input_folder, output_folder, target_width, target_height)

  print("done.")

  # input_folder = "/home/darkest/Downloads/FloodNet-Supervised_v1.0/test/masks"
  # output_folder = "/home/darkest/Downloads/FloodNet-Supervised_v1.0/test/masks-colored"
  # recolor_masks_in_folder(input_folder, output_folder)
