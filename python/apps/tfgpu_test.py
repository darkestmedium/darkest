import tensorflow as tf # TensorFlow registers PluggableDevices here.


devices = tf.config.list_physical_devices()
print(devices)

# list_phys_devices = tf.config.list_physical_devices()  # APU device is visible to TensorFlow.
# # [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:APU:0', device_type='APU')]
# print(list_phys_devices)


# a = tf.random.normal(shape=[5], dtype=tf.float32)  # Runs on CPU.
# b =  tf.nn.relu(a)         # Runs on APU.

# with tf.device("/APU:0"):  # Users can also use 'with tf.device' syntax.
#   c = tf.nn.relu(a)        # Runs on APU.

# with tf.device("/CPU:0"):
#   c = tf.nn.relu(a)        # Runs on CPU.

# @tf.function  # Defining a tf.function
# def run():
#   d = tf.random.uniform(shape=[100], dtype=tf.float32)  # Runs on CPU.
#   e = tf.nn.relu(d)        # Runs on APU.

# run()  # PluggableDevices also work with tf.function and graph mode.
