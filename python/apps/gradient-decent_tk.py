import os
import tensorflow as tf
import matplotlib.pyplot as plt

# For reproducibility
tf.random.set_seed(41)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


# %matplotlib inline
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (15, 8)


# Generating y = mx + c + random noise.
num_data = 1000

# True values of m and c
m_line = 3.3
c_line = 5.3


# Input (Generate random data between [-5,5]).
x = tf.random.uniform([num_data], minval=-5, maxval=5)

# Output (Generate data assuming y = mx + c + noise).
y_label = m_line * x + c_line + tf.random.normal(x.shape).numpy()
y = m_line * x + c_line

# Plot the generated data points. 
plt.plot(x, y_label, '.', color='g', label="Data points")
plt.plot(x, y, color='b', label='y = mx + c', linewidth=3)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show




def gradient_wrt_m_and_c(inputs, labels, m, c, k):
  """All arguments are defined in the training section of this notebook. 
  This function will be called from the training section.  
  So before completing this function go through the whole notebook.
  
  inputs (torch.tensor): input (X)
  labels (torch.tensor): label (Y)
  m (float): slope of the line
  c (float): vertical intercept of line
  k (torch.tensor, dtype=int): random index of data points
  """
  # gradient w.r.t to m is g_m 
  # gradient w.r.t to c is g_c
  
  ###
  ### YOUR CODE HERE
  ###
  # Calculate predicted values
  predicted_values = m * inputs + c

  # Calculate residuals (difference between predicted and actual values)
  residuals = predicted_values - labels

  # Calculate gradient w.r.t to m
  g_m = -2 * tf.reduce_sum(tf.gather(residuals, k) * tf.gather(inputs, k))

  # Calculate gradient w.r.t to c
  g_c = -2 * tf.reduce_sum(tf.gather(residuals, k))

  return g_m, g_c


X = tf.convert_to_tensor([-0.0374,  2.6822, -4.1152])
Y = tf.convert_to_tensor([ 5.1765, 14.1513, -8.2802])
m = 2
c = 3
k = tf.convert_to_tensor([0, 2])

gm, gc = gradient_wrt_m_and_c(X, Y, m, c, k)

print(f'Gradient of m : {gm:.2f}')
print(f'Gradient of c : {gc:.2f}')




def update_m_and_c(m, c, g_m, g_c, lr):
  """
  All arguments are defined in the training section of this notebook. 
  This function will be called from the training section.  
  So before completing this function go through the whole notebook.
  
  g_m = gradient w.r.t to m
  c_m = gradient w.r.t to c
  """
  # Update m and c parameters.
  # store updated value of m is updated_m variable
  # store updated value of c is updated_c variable
  ###
  ### YOUR CODE HERE
  ###
  updated_m = m - lr * g_m
  updated_c = c - lr * g_c

  return updated_m, updated_c



m = 2
c = 3
g_m = -24.93
g_c = 1.60
lr = 0.001
m, c = update_m_and_c(m, c, g_m, g_c, lr)

print('Updated m: {0:.2f}'.format(m))
print('Updated c: {0:.2f}'.format(c))




# Training







