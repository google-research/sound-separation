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
"""Network layers."""

import copy

import tensorflow.compat.v1 as tf

from . import network_config
from . import signal_util


class LayerNormalizationScalarParams(tf.keras.layers.LayerNormalization):
  """tf.keras.layers.LayerNormalization with scalar bias and scale params."""

  def __init__(self, param_shape=None, **kwargs):
    super(LayerNormalizationScalarParams, self).__init__(**kwargs)

  def build(self, input_shape):
    super(LayerNormalizationScalarParams, self).build([1] * len(input_shape))

  def call(self, inputs):
    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)

    if not self._fused:
      # Calculate the moments on the last axis (layer activations).
      mean, variance = tf.nn.moments(inputs, self.axis, keep_dims=True)

      scale, offset = self.gamma, self.beta

      # Compute layer normalization using the batch_normalization function.
      outputs = tf.nn.batch_normalization(
          inputs,
          mean,
          variance,
          offset=offset,
          scale=scale,
          variance_epsilon=self.epsilon)
    else:
      # Collapse dims before self.axis, and dims in self.axis
      pre_dim, in_dim = (1, 1)
      axis = sorted(self.axis)
      tensor_shape = tf.shape(inputs)
      for dim in range(0, ndims):
        dim_tensor = tensor_shape[dim]
        if dim < axis[0]:
          pre_dim = pre_dim * dim_tensor
        else:
          assert dim in axis
          in_dim = in_dim * dim_tensor

      squeezed_shape = [1, pre_dim, in_dim, 1]
      # This fused operation requires reshaped inputs to be NCHW.
      data_format = 'NCHW'

      inputs = tf.reshape(inputs, squeezed_shape)

      def _set_const_tensor(val, dtype, shape):
        return tf.fill(shape, tf.constant(val, dtype=dtype))

      # self.gamma and self.beta have the wrong shape for fused_batch_norm, so
      # we cannot pass them as the scale and offset parameters. Therefore, we
      # create two constant tensors in correct shapes for fused_batch_norm and
      # later contuct a separate calculation on the scale and offset.
      scale = _set_const_tensor(1.0, inputs.dtype, [pre_dim])
      offset = _set_const_tensor(0.0, inputs.dtype, [pre_dim])

      # Compute layer normalization using the fused_batch_norm function.
      outputs, _, _ = tf.nn.fused_batch_norm(
          inputs,
          scale=scale,
          offset=offset,
          epsilon=self.epsilon,
          data_format=data_format)

      outputs = tf.reshape(outputs, tensor_shape)

      scale, offset = self.gamma, self.beta

      if scale is not None:
        outputs = outputs * scale
      if offset is not None:
        outputs = outputs + offset

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs


def prelu(x, scope=None, init_param=0.1):
  """Parametric ReLU activation function."""
  with tf.variable_scope(name_or_scope=scope, default_name='prelu'):
    alpha = tf.get_variable('prelu', shape=[],
                            dtype=x.dtype,
                            initializer=tf.constant_initializer(init_param))
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def scale_layer(x, scope=None, init_param=1.0):
  """Scaling with a trainable scalar."""
  with tf.variable_scope(name_or_scope=scope, default_name='scale_layer'):
    alpha = tf.get_variable('scale_layer', shape=[],
                            dtype=x.dtype,
                            initializer=tf.constant_initializer(init_param))
    return alpha * x


def scalar_bias_layer(x, scope=None, init_param=0.0):
  """Add a trainable scalar bias variable."""
  with tf.variable_scope(name_or_scope=scope, default_name='scale_layer'):
    bias = tf.get_variable('scalar_bias_layer', shape=[],
                           dtype=x.dtype,
                           initializer=tf.constant_initializer(init_param))
    return x + bias


def update_config_from_kwargs(in_config, **kwargs):
  """Clones and overwrites attributes from keyword arguments.

  The variables that are going to be overwritten must be indicated by setting
  their values to be -99 for numeric variables or "external" for string
  variables. If this rule is not followed, an error is raised.
  Boolean variables cannot be overwritten.

  Args:
    in_config: Input object configuration.
    **kwargs: Keyword arguments to overwrite attributes of object.
  Returns:
    An updated object.
  """
  config = copy.deepcopy(in_config)
  for item, value in kwargs.items():
    if hasattr(config, item):
      if isinstance(value, list):
        raise ValueError('Cannot update repeated fields using a list.')
      else:
        value_to_replace = getattr(config, item)
        if (isinstance(value_to_replace, bool) or
            (isinstance(value_to_replace, (int, float)) and
             value_to_replace != -99) or
            (isinstance(value_to_replace, type('')) and
             value_to_replace != 'external')):
          if value != value_to_replace:
            raise ValueError(
                'Overwriting {} that was equal to {} '
                'with {} not allowed. Make sure that the object value to be '
                'overwritten is equal to -99 for integer and floats and is '
                'equal to \'external\' for strings. Booleans are not allowed '
                'to be overwritten. If you think you need to overwrite them, '
                'consider changing them to strings or '
                'try and not to overwrite those variables in your '
                'configuration or python code, for example by defining '
                'separate object for each type.'
                ''.format(item, value_to_replace, value))
        setattr(config, item, value)
    else:
      raise ValueError('Updating dictionary contains a key {} that does not '
                       'exist in the config (object).'.format(item))
  tf.logging.debug('\n----------\nbefore change: ' + str(in_config) +
                   '\n----------\nafter change: ' + str(config))
  return config


def copy_attributes_from_object(in_config, in_object, attr_list):
  """Clones a object and copies attributes from an object to it."""
  config = copy.deepcopy(in_config)
  for item in attr_list:
    if hasattr(in_object, item):
      value = getattr(in_object, item)
    else:
      raise ValueError('attr_list contains an attribute {} that does not '
                       'exist in the updating object.'.format(item))
    if hasattr(config, item):
      if isinstance(value, list):
        getattr(config, item).extend(value)
      else:
        setattr(config, item, value)
    else:
      raise ValueError('attr_list contains an attribute {} that does not '
                       'exist in the config (object).'.format(item))
  tf.logging.debug('\n----------\nbefore change: ' + str(in_config) +
                   '\n----------\nafter change: ' + str(config))
  return config


def dense_layer(x, config):
  """Configurable dense layer.

  Supports scale layer.
  Args:
    x: Input tensorflow tensor.
    config: network_config.DenseLayer object.
  Returns:
    An output tensor.
  """
  if config.num_outputs < 0:
    raise ValueError('Please set num_outputs to a positive number:'
                     ' {}'.format(config.num_outputs))
  if config.use_bias:
    bias_initializer = tf.zeros_initializer
  else:
    bias_initializer = None
  dense_output = tf.keras.layers.Dense(
      config.num_outputs,
      activation=get_activation_fn(config.activation),
      use_bias=config.use_bias,
      kernel_initializer=config.kernel_initializer,
      bias_initializer=bias_initializer,
      name='dense')(x)
  if config.scale >= 0.0:
    dense_output = scale_layer(dense_output,
                               scope='dense_scale',
                               init_param=config.scale)
  if config.add_scalar_bias:
    dense_output = scalar_bias_layer(dense_output, scope='dense_scalar_bias',
                                     init_param=0.0)
  return dense_output


def norm_fn_from_type(norm_type, **kwargs):
  config = network_config.NormLayer()
  config.norm_type = norm_type
  return norm_fn(config, **kwargs)


def norm_fn(config, **kwargs):
  """Returns a normalization function/layer.

  Args:
    config: network_config.NormLayer object.
    **kwargs: Keyword arguments to overwrite config entries.
  Returns:
    A normalization function that acts on a single input tensor and
    returns another tensor with the same dimensions, the output tensor is
    a normalized version of the input tensor according to the configuration
    parameters.
  """
  config = update_config_from_kwargs(config, **kwargs)
  if config.norm_type == 'global_layer_norm':
    reduction_axes = [config.time_axis, config.bin_axis]
  elif config.norm_type == 'instance_norm':
    reduction_axes = [config.time_axis]
  elif config.norm_type == 'layer_norm':
    reduction_axes = [config.bin_axis]
  else:
    raise ValueError('Unknown norm layer type: ' + config.norm_type)
  norm_fn_out = LayerNormalizationScalarParams(
      axis=reduction_axes,
      name=config.norm_type)
  return norm_fn_out


def norm_layer(x, config):
  """Normalization layer.

  Args:
    x: Input tensor.
    config: network_config.NormLayer object.
  Returns:
    An output tensor.
  """
  fn = norm_fn(config)
  return fn(x)


def time_convolution_layer(x, config):
  """Time convolution layer with possible post-pooling.

  Args:
    x: Input tensor.
    config: network_config.TimeConvLayer object.
  Returns:
    An output tensor.
  """
  dilation_rate = [config.dilation, 1]
  kernel_size = [config.kernel_size, 1]
  stride = [config.stride, 1]
  if config.separable:
    y = tf.keras.layers.DepthwiseConv2D(
        kernel_size,
        padding='SAME',
        depth_multiplier=1,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=None,
        name='separable_conv',
        )(x)
  else:
    filters = tf.shape(x)[-1]
    y = tf.layers.Conv2D(
        filters, kernel_size,
        stride=stride,
        padding='SAME',
        rate=dilation_rate,
        activation_fn=None,
        scope='2d_conv',
        )(x)
  return y


def norm_and_activation_layer(x, config):
  """Normalization and activation layers.

  Args:
    x: Input tensor.
    config: network_config.NormAndActivationLayer object.
  Returns:
    An output tensor.
  """
  activation_fn = get_activation_fn(config.activation)
  if config.norm_after_act:
    x = activation_fn(x) if activation_fn else x
    x = norm_layer(x, config.norm_layer)
  else:
    x = norm_layer(x, config.norm_layer)
    x = activation_fn(x) if activation_fn else x
  return x


def tdcn_block(x, config):
  """Runs a TDCN block according to config.

  Args:
    x: Input tensor.
    config: network_config.TDCNBlock object.
  Returns:
    An output tensor
  """
  with tf.variable_scope('tdcn_block'):
    dense1_config = update_config_from_kwargs(
        config.dense1, num_outputs=config.num_conv_channels,
        activation='linear')
    with tf.variable_scope('dense1'):
      y = dense_layer(x, dense1_config)

    normact1_config = update_config_from_kwargs(
        config.normact1, activation=config.middle_activation)
    with tf.variable_scope('normact1'):
      y = norm_and_activation_layer(y, normact1_config)

    timeconv_config = update_config_from_kwargs(
        config.tclayer, dilation=config.dilation, stride=config.stride,
        separable=config.separable, kernel_size=config.kernel_size)
    with tf.variable_scope('timeconv'):
      z = time_convolution_layer(y, timeconv_config)

    normact2_config = update_config_from_kwargs(
        config.normact2, activation=config.middle_activation)
    with tf.variable_scope('normact2'):
      z = norm_and_activation_layer(z, normact2_config)

    dense2_config = update_config_from_kwargs(
        config.dense2, num_outputs=config.bottleneck, scale=config.scale,
        activation=config.end_of_block_activation)
    with tf.variable_scope('dense2'):
      z = dense_layer(z, dense2_config)

    # Add residual connection from input x.
    if config.resid:
      z = tf.add(x[..., :config.bottleneck], z)
  return z


def _find_scale_function(scale_type):
  """Returns a function that calculates a scale for a block b."""
  if scale_type == 'exponential':
    scale_fn = lambda b: 0.9**b
  elif scale_type == 'none':
    scale_fn = lambda b: -1.0  #  Negative disables scaling in dense_layer
  else:
    raise ValueError('Unknown scale type {}'.format(scale_type))
  return scale_fn


def improved_tdcn(input_activations, config):
  """Creates improved time-dilated convolution network (TDCN++) from [1].

  [1] Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson,
      Jonathan Le Roux, John R. Hershey, "Universal Sound Separation,"
      https://arxiv.org/abs/1905.03330.
  Total number of convolutional layers is num_conv_blocks * num_repeats.

  Args:
    input_activations: Tensor (batch_size, num_frames, mics x depth, bins)
      of mixture input spectrograms.
    config: network_config.ImprovedTDCN object.
  Returns:
    layer_activations: activations of the last convolution of shape
        (batch_size, num_frames, bottleneck x mics x depth).
  """
  batch_size = signal_util.static_or_dynamic_dim_size(input_activations, 0)
  num_frames = signal_util.static_or_dynamic_dim_size(input_activations, 1)
  mics_and_depth = signal_util.static_or_dynamic_dim_size(input_activations, 2)

  layer_activations = input_activations
  # layer_activations is shape (batch_size, num_frames, mics x depth, bins).

  initial_dense_config = update_config_from_kwargs(
      config.initial_dense_layer,
      num_outputs=config.prototype_block[0].bottleneck)
  with tf.variable_scope('initial_dense'):
    layer_activations = dense_layer(layer_activations, initial_dense_config)

  input_of_block = []
  num_blocks = len(config.block_prototype_indices)
  find_scale_fn = _find_scale_function(config.scale_tdcn_block)
  with tf.variable_scope('improved_tdcn'):
    for block in range(num_blocks):
      proto_block = config.block_prototype_indices[block]
      connections_to_block = []
      for src, dest in zip(config.skip_residue_connection_from_input_of_block,
                           config.skip_residue_connection_to_input_of_block):
        if dest == block:
          assert src < dest
          connections_to_block.append(src)
      for prev_block in connections_to_block:
        residue_dense_config = update_config_from_kwargs(
            config.residue_dense_layer,
            num_outputs=config.prototype_block[proto_block].bottleneck)
        with tf.variable_scope('res_dense_{}_to_{}'.format(prev_block, block)):
          layer_activations += dense_layer(input_of_block[prev_block],
                                           residue_dense_config)
      input_of_block.append(layer_activations)
      scale_tdcn_block = find_scale_fn(block)
      # conv_inputs is shape
      # (batch_size, num_frames, mics x depth, bottleneck).

      with tf.variable_scope('conv_block_%d' % block):
        dilation = config.block_dilations[block]
        tdcn_block_config = config.prototype_block[proto_block]
        tdcn_block_config = update_config_from_kwargs(
            tdcn_block_config,
            scale=scale_tdcn_block,
            dilation=dilation)

        layer_activations = tdcn_block(
            layer_activations,
            tdcn_block_config)
    # layer_activations is of shape
    # (batch_size, num_frames, mics x depth, output_size).
    num_frames = signal_util.static_or_dynamic_dim_size(layer_activations, 1)
    # output_size = bottleneck when resid=true, concat_input=false.
    # output_size = num_blocks * bottleneck + num_coeff when
    #   resid=false, concat_input=true.
    output_size = signal_util.static_or_dynamic_dim_size(layer_activations, -1)
    layer_activations = tf.reshape(
        layer_activations,
        (batch_size, num_frames,
         mics_and_depth * output_size))
    # layer_activations is now shape
    # (batch_size, num_frames, mics x depth x bottleneck).
  return layer_activations


def get_activation_fn(name):
  """Returns an activation function."""
  act_fns = {
      'sigmoid': tf.nn.sigmoid,
      'relu': tf.nn.relu,
      'leaky_relu': tf.nn.leaky_relu,
      'tanh': tf.nn.tanh,
      'prelu': prelu,
      'linear': None,
  }
  if name not in act_fns:
    raise ValueError('Unsupported activation %s' % name)
  return act_fns[name]
