# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow signal manipulation utilities."""

import numpy as np
import tensorflow.compat.v1 as tf


def static_or_dynamic_dim_size(tensor, i):
  """Static size for dimension `i` if available, otherwise dynamic size."""
  static_shape = tensor.shape
  dyn_shape = tf.shape(tensor)
  return (static_shape[i].value if hasattr(static_shape[i], 'value')
          else static_shape[i]) or dyn_shape[i]


def smart_shape(tensor):
  """Shape of tensor with static and/or dynamic dimensions.

  Args:
    tensor: A tf.Tensor.

  Returns:
    A list containing static (type int) and dynamic (tf.Tensor) dim sizes.
  """
  dims = []
  for i in range(len(tensor.shape)):
    dims.append(static_or_dynamic_dim_size(tensor, i))
  return tuple(dims)


def enclosing_power_of_two(value):
  """Return 2**N for smallest integer N such that 2**N >= value."""
  return int(2**np.ceil(np.log2(value)))


def stacked_real_imag_abs(values, offset=1e-8):
  return tf.sqrt(tf.math.square(values[..., 0]) +
                 tf.math.square(values[..., 1]) + offset)


def stabilized_real_imag_abs(real_values, imag_values, offset=1e-8):
  """Outputs stabilized absolute of real_values+j*imag_values."""
  return tf.sqrt(real_values**2 + imag_values**2 + offset)


def stabilized_power_compress_abs(values, power=0.5, offset=1e-8):
  """Outputs stabilized power-law compression of the abs of the input."""
  if values.dtype is tf.complex64:
    # Note that tf.abs(a+bj) = tf.sqrt(a*a+b*b).
    # Need to avoid 0.0 for complex numbers.
    # The offset is in default magnitude-level offset. We need to square
    # it when it is used for power-level offset. However, (1e-8)**2=1e-16
    # in default could be too much small, here we use offset**1.5 as the
    # power-level offset.
    stabilized_values = stabilized_real_imag_abs(tf.real(values),
                                                 tf.imag(values),
                                                 offset=offset**1.5)
  else:
    stabilized_values = tf.abs(values) + offset
  return stabilized_values if power == 1.0 else tf.pow(
      stabilized_values, power)


def stabilized_log_base(x, base=10., stabilizer=1e-8):
  logx = tf.log(x + stabilizer)
  logb = tf.log(tf.constant(base, dtype=logx.dtype))
  return logx / logb


def make_argmax_indices(input_tensor, axis=1):
  """Given a tensor, find argmax over the axis and return indices.

  The output can be used as indices in tf.gather_nd and tf.scatter_nd ops.
  Args:
    input_tensor: A tensor to perform argmax over.
    axis: Which axis to take argmax over.
  Returns:
    indices: An index tensor that can be used in tf.gather_nd and
      tf.scatter_nd ops to gather from and scatter to the max index.
  """
  extreme_idx = tf.argmax(input_tensor, axis=axis, output_type=tf.int32)
  # tf.argmax does not have keepdims argument, so we do it separately.
  extreme_idx = tf.expand_dims(extreme_idx, axis=axis)
  in_shape = tf.shape(extreme_idx)
  in_rank = len(extreme_idx.shape.as_list())
  idx_list = []
  for dim in range(in_rank):
    if dim != axis:
      dim_len = in_shape[dim]
      pre_broadcast_shape = [1] * in_rank
      pre_broadcast_shape[dim] = dim_len
      dim_idx = tf.reshape(tf.cast(tf.range(dim_len), extreme_idx.dtype),
                           pre_broadcast_shape)
      idx_list.append(tf.broadcast_to(dim_idx, in_shape))
    else:
      idx_list.append(extreme_idx)
  indices = tf.stack(idx_list, axis=-1)
  return indices
