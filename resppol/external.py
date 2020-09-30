# !/usr/bin/env python

import logging as log
import time

try:
    import tensorflow as tf
except:
    print("Could not import TensorFlow, use Numpy instead.")


def build_tensor(np_arr, dtype=tf.float32):
	tf_tensor = tf.convert_to_tensor(np_arr, dtype=dtype)
	return tf_tensor


def tensor_solver(tensor_1, tensor_2, device_name="/device:GPU:0", name=None):
	with tf.device(device_name):
		start = time.perf_counter()
		results = tf.linalg.solve(tensor_1, tensor_2, adjoint=False, name=name)
		results_reshape = tf.reshape(results, (tensor_1.shape[0], ))
		finish = time.perf_counter()
		log.debug(f"Solving matrix with TF took {(finish-start)*1000:0.2f} ms.")
		return results_reshape
