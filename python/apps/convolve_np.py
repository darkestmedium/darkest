from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import numpy as np




def convolve(
  input_array: np.ndarray,
  kernel: np.ndarray,
  verbose: bool = False,
) -> np.ndarray:

  """
  convolve

  Parameters

  input_array: Array on which convolution is done.
  output_image: Placeholder array for output.
  kernel: Kernel/ Filter.
  
  Returns

  Convolved array.
  """

  inshape = input_array.shape
  kshape = kernel.shape

  output_array = np.zeros((inshape[0] - kshape[0] + 1, inshape[1] - kshape[1] + 1))

  for i in range(0, input_array.shape[0]):
    for j in range(0, input_array.shape[1]):
      try:
        ###
        ### YOUR CODE HERE
        ###
        # Extract the region of the input_array for convolution
        region = input_array[i:i+kshape[0], j:j+kshape[1]]
        
        # Perform element-wise multiplication and sum
        convolution = np.sum(region * kernel)
        
        # Assign the result to the output_array
        output_array[i, j] = convolution

      except Exception as exception:
        if verbose:
          print("WARN: Image boundary is ignored.")
          raise exception

  return output_array.astype(np.int32)


kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

input_array = np.array([[7,2,3,3,8],[4,5,3,8,4],[3,3,2,8,4],[2,8,7,2,7],[5,4,4,5,4]])
ouput_array = convolve(input_array, kernel)

print(f"Output array:")
print(ouput_array)
