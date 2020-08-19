import tensorflow as tf
import os

_current_path = os.path.dirname(os.path.realpath(__file__))
_tf_graphics_module = tf.load_op_library(os.path.join(_current_path, 'custom_graphics.so'))
decode_block = _tf_graphics_module.decode_block

