import cv2
import os
import shutil




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
