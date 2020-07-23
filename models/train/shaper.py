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
"""Class for checking and changing shape of tensors with semantic axis names."""

import numpy as np

import tensorflow.compat.v1 as tf

from . import signal_util


class Shaper(object):
  """Class to change shape of tensors with meaningful axis names.

  The main goal of this class is to simplify shaping tf.Tensors using reshapes
  and transposes, while also providing implicit documentation of axis labels.
  Main methods are change, which changes the shape of any tensor given a list of
  input and output axis names or unit values (1), and register_axes, which both
  checks and registers axis names with their corresponding dim sizes from an
  input tensor.

  For example, a tf.Tensor x of shape (batch, mic, 1, time) can be changed to
  shape (batch, time * mic, 1, 1) using the following:
      shaper = signal_util.Shaper()
      x_new = signal_util.change(x, ['batch', 'mic', 1, 'time'],
                                 ['batch', ('time', 'mic'), 1, 1])
  which corresponds to the equivalent TensorFlow calls:
      x = tf.squeeze(x, 2)
      # x is now shape (batch, mic, time).
      x = tf.transpose(x, [0, 2, 1])
      # x is now shape (batch, time, mic).
      x = tf.reshape(xew, (batch, time * mic))
      # x is now shape (batch, time * mic).
      x = tf.expand_dims(x, 2)
      # x is now shape (batch, time * mic, 1).
      x = tf.expand_dims(x, 3)
      # x is now shape (batch, time * mic, 1, 1).
      x_new = x

  Notice that using Shaper, the comment about x_new's shape is unnecessary.
  Shaper also enables checking of tensor shape, since Shaper keeps track of the
  axis names and the corresponding sizes observed since its creation. To check
  the shape, register_axes can be used. For example, if a tensor x is shape
  (batch, time * mic), the following call would raise a ValueError exception::
      shaper.register_axes(x, (time, batch, mic))
  while the following call would not raise an exception:
      shaper.register_axes(x, [batch, (time, mic)])
  This register_axes call also serves to document the assumed shape of x.

  Attributes:
    axis_sizes: A dict mapping axis names (strings) to dim sizes (ints or
        tf.Tensors), for all the tf.Tensors this object has seen through
        register_axes or change calls.
  """

  def __init__(self, axis_sizes=None):
    """Inits Shaper class with optional axis_sizes dict."""
    self.axis_sizes = axis_sizes if axis_sizes else {}

  def _flatten_axes(self, axes):
    """Constructs flattened version of axes.

    Args:
      axes: A list containing strings and tuples of strings. For example,
          [batch, (mic, time)] represents a Tensor of shape (batch, mic * time).

    Returns:
      axes_flat: List of ints (static dims) and/or tf.Tensors (dynamic dims).
    """
    # Convert strings to 1-tuples.
    axes_flat = [a if isinstance(a, tuple) else (a,) for a in axes]
    # Concatenate the tuples.
    return list(sum(axes_flat, ()))

  def _register_axis_sizes(self, tensor, axes):
    """Update internal axis_sizes dict given a tensor and axes specification.

    Args:
      tensor: tf.Tensor to check the shape of.
      axes: List of strings and/or tuples of strings specifying assumed axes.

    Raises:
      ValueError: If the numerical size of a tensor's dim doesn't match the
          expected size for the axis name.
      ValueError: If any axis name in a packed dim is not stored in axis_sizes.
    """
    for dim, axis in enumerate(axes):
      if isinstance(axis, tuple):
        # Multiple axes packed together into one dim.
        for name in axis:
          if name not in self.axis_sizes:
            raise ValueError(
                'Axis "{}" not known, set its size first.'.format(name))
      else:
        # This dim is not packed.
        if axis in self.axis_sizes:
          size = signal_util.static_or_dynamic_dim_size(tensor, dim)
          if not tf.is_tensor(size) and size != self.axis_sizes[axis]:
            raise ValueError('Incorrect axis name "{axis}" for dim {dim}: got '
                             '{size} but expected {expected_size}.'.format(
                                 axis=axis, dim=dim, size=size,
                                 expected_size=self.axis_sizes[axis]))
        else:
          self.axis_sizes[axis] = signal_util.static_or_dynamic_dim_size(tensor,
                                                                         dim)

  def _get_transpose_arg(self, input_axes_flat, output_axes_flat):
    """Returns permutation to convert input axes to output axes.

    Args:
      input_axes_flat: A list of strings, corresponding to input axes order.
      output_axes_flat: A list of strings, corresponding to output axes order.

    Returns:
      A list that can be used as the perm arg of tf.transpose to permute
      input_axes_flat to output_axes_flat.

    Raises:
      ValueError: output_axes_flat is not a permutation of input_axes_flat.
    """
    if set(input_axes_flat) != set(output_axes_flat):
      raise ValueError('output_axes_flat of {} is not a permutation of {}.'
                       .format(output_axes_flat, input_axes_flat))
    transpose_arg = []
    for output_axis in output_axes_flat:
      transpose_arg.append(input_axes_flat.index(output_axis))
    return transpose_arg

  def _transpose(self, tensor, input_axes_flat, output_axes_flat):
    """Transposes tensor from input axes to output axes.

    Args:
      tensor: A tf.Tensor to transpose.
      input_axes_flat: A list of strings, corresponding to input axes order.
      output_axes_flat: A list of strings, corresponding to output axes order.

    Returns:
      A transposed tf.Tensor.
    """
    transpose_arg = self._get_transpose_arg(input_axes_flat, output_axes_flat)
    if transpose_arg != range(len(input_axes_flat)):
      return tf.transpose(tensor, transpose_arg)
    else:
      return tensor

  def _shape_from_axes(self, axes):
    """Get numerical shape from axes specification (packed or unpacked).

    Args:
      axes: List of strings and/or tuples of strings specifying assumed axes.

    Returns:
      A list of ints and tf.Tensors providing the numerical shape using the
      stored axis_sizes.
    """
    shape = []
    for axis in axes:
      if isinstance(axis, tuple):
        shape.append(np.prod([self.axis_sizes[aa] for aa in axis]))
      else:
        shape.append(self.axis_sizes[axis])
    return shape

  def _separate_unit_axes(self, axes):
    """Separates axes into unit and non-unit axes.

    Args:
      axes: List of string, tuple of string or 1 valued axes.

    Returns:
      Tuple of:
        - unit_axes = indices of unit values in axis.
        - non_unit_axes = values that are non-unit.
    """
    unit_axes = [idx for idx, axis in enumerate(axes) if axis == 1]
    non_unit_axes = [axis for axis in axes if axis != 1]

    return (unit_axes, non_unit_axes)

  def update_axis_sizes(self, new_axis_sizes):
    """Update internal dict mapping axis name to size.

    Args:
      new_axis_sizes: A dict mapping axis names (strings) to dim sizes (ints or
          tf.Tensors).

    Raises:
      ValueError: A conflicting value provided for an axis already stored.
    """
    for name, size in new_axis_sizes.items():
      if name in self.axis_sizes and size != self.axis_sizes[name]:
        raise ValueError('Conflicting value provided for axis "{}": {} instead '
                         'of {}'.format(name, size, self.axis_sizes[name]))
    self.axis_sizes.update(new_axis_sizes)

  def register_axes(self, tensor, axes):
    """Register axis names and their sizes, given a tensor and axes spec.

    Also checks consistency of axis names that have already been recorded.

    Args:
      tensor: tf.Tensor to check the shape of.
      axes: List of strings and/or tuples of strings specifying assumed axes.

    Raises:
      ValueError: If the numerical size of a tensor's dim doesn't match the
          expected size for the axis name.
      ValueError: If any axis name in a packed dim is not stored in axis_sizes.
      ValueError: If the length of the provided axes spec is different than the
          length of the tensor's shape.
    """
    if len(axes) != len(tensor.shape):
      raise ValueError('axes spec of {} has different len than tensor\'s shape'
                       '{}'.format(axes, tensor.shape))

    unit_axes, non_unit_axes = self._separate_unit_axes(axes)
    if unit_axes:
      tensor = tf.squeeze(tensor, unit_axes)

    self._register_axis_sizes(tensor, non_unit_axes)

  def change(self, tensor, input_axes, output_axes):
    """Changes the shape of a tensor.

    Args:
      tensor: tf.Tensor to change the shape of.
      input_axes: List of strings, tuples of strings, and unit values (1),
        specifying input axes.
      output_axes: List of strings, tuples of strings, and unit values (1),
        specifying output axes.

    Returns:
      tf.Tensor with changed shape.

    Raises:
      ValueError: If input and output axes contain different axis names.
      ValueError: If the numerical size of a tensor's dim doesn't match the
          expected size for the axis name.
      ValueError: If any axis name in a packed dim is not stored in axis_sizes.
    """
    unit_input_axes, input_axes = self._separate_unit_axes(input_axes)
    unit_output_axes, output_axes = self._separate_unit_axes(output_axes)
    if unit_input_axes:
      tensor = tf.squeeze(tensor, unit_input_axes)

    input_axes_flat = self._flatten_axes(input_axes)
    self._register_axis_sizes(tensor, input_axes)
    packed_input = any([len(a) > 1 for a in input_axes])

    output_axes_flat = self._flatten_axes(output_axes)
    packed_output = any([len(a) > 1 for a in output_axes])

    if set(input_axes_flat) != set(output_axes_flat):
      raise ValueError('Input and output shapes must contain same (non-unit)'
                       ' axes, are {} and {}'.format(input_axes, output_axes))

    if len(input_axes) < len(output_axes):
      # Unpacking input to output.
      new_shape = self._shape_from_axes(input_axes_flat)
      tensor_new = tf.reshape(tensor, new_shape)
      tensor_new = self._transpose(tensor_new, input_axes_flat,
                                   output_axes_flat)

    elif len(input_axes) > len(output_axes):
      # Packing input to output.
      tensor_new = self._transpose(tensor, input_axes_flat, output_axes_flat)
      new_shape = self._shape_from_axes(output_axes)
      tensor_new = tf.reshape(tensor_new, new_shape)

    else:
      tensor_new = tensor

      if packed_input:
        # Unpack the input
        new_shape = self._shape_from_axes(input_axes_flat)
        tensor_new = tf.reshape(tensor_new, new_shape)

      tensor_new = self._transpose(tensor_new, input_axes_flat,
                                   output_axes_flat)

      if packed_output:
        # Pack the output.
        new_shape = self._shape_from_axes(output_axes)
        tensor_new = tf.reshape(tensor_new, new_shape)

    for axis in unit_output_axes:
      tensor_new = tf.expand_dims(tensor_new, axis)

    return tensor_new
