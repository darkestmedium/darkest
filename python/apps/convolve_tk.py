from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow as tf






def convolve(
  in_tensor: tf.Tensor,
  kernel: tf.Tensor,
  verbose: bool = False,
) -> tf.Tensor:

  """
  convolve

  Parameters

  input_array: Array on which convolution is done.
  output_image: Placeholder array for output.
  kernel: Kernel/ Filter.
  
  Returns

  Convolved array.
  """
  
  inshape = in_tensor.shape
  kshape = kernel.shape
  
  output_array = tf.zeros((inshape[0] - kshape[0] + 1, inshape[1] - kshape[1] + 1), dtype=tf.int32)

  for i in range(0, inshape[0]):
    for j in range(0, inshape[1]):
      try:
        ###
        ### YOUR CODE HERE
        ###
        # Extract the subarray of input tensor
        region = in_tensor[i:i + kshape[0], j:j + kshape[1]]
        # Calculate the convolution
        convolution = tf.reduce_sum(region * kernel)
        # Update the output array
        output_array = tf.tensor_scatter_nd_update(output_array, [[i, j]], [convolution])
      except Exception as exception:
        if verbose:
          print("Warn: Image boundary is ignored.")
          raise exception

  return output_array



kernel = tf.constant([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

input_array = tf.constant([[7,2,3,3,8],[4,5,3,8,4],[3,3,2,8,4],[2,8,7,2,7],[5,4,4,5,4]])
ouput_array = convolve(input_array, kernel)

print(f"Output array:")
print(ouput_array)