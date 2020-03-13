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
"""Configuration for network architectures."""

import copy
import attr
import typing


@attr.attrs
class NormLayer(object):
  """Normalization layer configurations."""
  # Options for norm_type string are:
  # 'global_layer_norm': layer normalization over time_axis and bin_axis.
  norm_type = attr.attrib(type=typing.Text, default='none')
  # If True, normalize each bin independently (do not reduce over bins)
  bin_wise = attr.attrib(type=bool, default=False)
  # time axis
  time_axis = attr.attrib(type=int, default=-3)
  # bin (frequency or time dependent feature vector) axis
  bin_axis = attr.attrib(type=int, default=-1)
  # activation_fn to apply after norm
  activation = attr.attrib(type=typing.Text, default='linear')


@attr.attrs
class NormAndActivationLayer(object):
  # norm layer to use
  norm_layer = attr.attrib(type=typing.Optional[NormLayer], default=None)
  # activation type to use
  activation = attr.attrib(type=typing.Text, default='prelu')
  # normalize after activation or reverse
  norm_after_act = attr.attrib(type=bool, default=True)


@attr.attrs
class DenseLayer(object):
  """Dense layer configurations."""
  # Length of output vector. Overwritten before python call typically.
  num_outputs = attr.attrib(type=int, default=-99)
  use_bias = attr.attrib(type=bool, default=False)
  # activation function to apply after dense layer. Options are:
  #   'sigmoid', 'relu', 'leaky_relu', 'tanh', 'prelu'
  #   'power_law', 'linear', 'parametrized_sigmoid'.
  activation = attr.attrib(type=typing.Text, default='linear')
  # Add a learnable scale parameter alpha which is applied as
  # y = alpha f(W x) where W is the dense matrix and f is the activation.
  # Initialize alpha = scale if scale >= 0 and make it learnable.
  # If scale < 0, no scale layer is added.
  scale = attr.attrib(type=float, default=-1.0)
  # Add scalar bias initialized with 0.
  # The scalar bias is added after all other operations.
  add_scalar_bias = attr.attrib(type=bool, default=False)
  # Initialization for kernel matrix.
  kernel_initializer = attr.attrib(type=typing.Text, default='glorot_uniform')


@attr.attrs
class TimeConvLayer(object):
  """Time convolution layer configurations."""
  # Filter length for time convolutions.
  kernel_size = attr.attrib(type=int, default=3)
  # Dilation value of time convolution.
  dilation = attr.attrib(type=int, default=-99)
  # Stride value of time convolution.
  stride = attr.attrib(type=int, default=1)
  # Whether the convolution block is separable or full which is a 2D conv.
  separable = attr.attrib(type=bool, default=True)


@attr.attrs
class TDCNBlock(object):
  """TDCN block configuration."""
  # First dense layer
  dense1 = attr.attrib(type=typing.Optional[DenseLayer], default=None)
  # Normalization and activation following dense1
  normact1 = attr.attrib(
      type=typing.Optional[NormAndActivationLayer], default=None)
  # Time convolution layer
  tclayer = attr.attrib(type=typing.Optional[TimeConvLayer], default=None)
  # Normalization and activation following tclayer
  normact2 = attr.attrib(
      type=typing.Optional[NormAndActivationLayer], default=None)
  # Second dense layer following normact2
  dense2 = attr.attrib(type=typing.Optional[DenseLayer], default=None)
  # Channels dimension at the input to each convolution block.
  # Overwrites dense2's num_outputs variable.
  bottleneck = attr.attrib(type=int, default=256)
  # Number of filters within each convolutional block.
  # Overwrites dense1's num_outputs variable.
  num_conv_channels = attr.attrib(type=int, default=512)
  # Filter length within each convolutional block.
  # Overwrites tclayer's kernel_size
  kernel_size = attr.attrib(type=int, default=3)
  # Dilation along time.
  # Overwrites tclayer's dilation, usually set from outside
  dilation = attr.attrib(type=int, default=-99)
  # Stride along time.
  # Overwrites tclayer's stride
  stride = attr.attrib(type=int, default=1)
  # Whether the conv2d block is separable or full.
  # overwrites separable variable of tclayer
  separable = attr.attrib(type=bool, default=True)
  # Activation function to use within the block.
  # Overwrites activation of normact1, normact2.
  middle_activation = attr.attrib(type=typing.Text, default='prelu')
  # Activation function to use at the end of the block.
  # Overwrites activation of dense2.
  end_of_block_activation = attr.attrib(type=typing.Text, default='linear')
  # Whether to use residual connections in each block.
  resid = attr.attrib(type=bool, default=True)
  # The value which initializes a scale parameter at the block's output.
  # Overwrites dense2's scale variable.
  scale = attr.attrib(type=float, default=-99)


@attr.attrs
class ImprovedTDCN(object):
  """TDCN++ Configuration.

  Note: in the config in this file, a composite config can contain
  sub-message config which describe sub-layers in a composite layer.
  The submessages define a prototypical sub-layer which can be reused multiple
  times where some of its variables can be overwritten by the calling
  composite layer. Hierarchy works as follows. A composite config may
  overwrite some of the variables of its own sub-messages during the
  python call. The variables that are going to be overwritten must be
  indicated by setting their values to be -99 for numeric variables or
  'external' for string variables. If this rule is not followed, an error
  is raised.

  A TDCN++ [2], inspired by [1], consists of a stack of dilated convolutional
  layers that predict a mask. An initial 1x1 convolution layer converts a shape
  (batch_size, ..., num_frames, num_coeffs) input into shape
  (batch_size, ..., num_frames, bottleneck). Then, there are `num_repeats`
  repeat modules stacked on top of each other. Within each repeat module, there
  are `num_conv_blocks` convolutional blocks, where the ith block has a
  dilation factor of 2^i. Each block consists of the following sequence: a
  dense layer with num_outputs of `num_conv_channels`, a leaky ReLU activation
  and normalization (normalization is specified by `norm`; also, the order of
  activation and normalization can be swapped by `norm_after_act`), a separable
  convolution across time with `num_conv_channels` filters of length
  `kernel_size`, a leaky ReLU activation and normalization,
  and a second dense layer with num_outputs of `bottleneck`.
  There is a residual connection from the input of each
  convolution block to its output.

  [1] Yi Luo, Nima Mesgarani, 'Conv-TasNet: Surpassing Ideal Time-Frequency
      Masking for Speech Separation,' https://arxiv.org/pdf/1809.07454.pdf.
  [2] Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson,
      Jonathan Le Roux, John R. Hershey, "Universal Sound Separation,"
      https://arxiv.org/abs/1905.03330.
  """
  # Initial dense layer applied before sending data to TDCN blocks.
  initial_dense_layer = attr.attrib(
      type=typing.Optional[DenseLayer], default=None)
  # Prototypical TDCN block. There may be multiple prototypical blocks and
  # they all can be used in the TDCN network.
  # In addition, dilations and scales will be changed by this parent
  # layer's python function when it calls each block's python function.
  prototype_block = attr.attrib(type=typing.List[TDCNBlock], factory=list)
  # Specify the prototype_block index for each block in the network.
  # The prototype block index is a 0-based index. The length of this array
  # is going to be interpreted as the number of blocks in the network.
  block_prototype_indices = attr.attrib(type=typing.List[int], factory=list)
  # Specify the dilation value for each block (integer >= 1)
  # Best setting for dilations is 2^i where i=0, ..., num_conv_blocks-1 for each
  # 'repeat' and this pattern must be repeated num_repeats times. Note that
  # there is no more a concept of 'repeat' in this config since dilations can be
  # arbitrary.
  # One can enable 'repeat's by defining dilations appropriately e.g.
  # block_dilations = [1,2,4,8,16,1,2,4,8,16,1,2,4,8,16]
  # corresponds to num_repeats=3 and num_conv_blocks=5.
  block_dilations = attr.attrib(type=typing.List[int], factory=list)
  # Add a skip residue connection from the input of block (0-based index),
  # each block input is passed through a dense layer before being added.
  skip_residue_connection_from_input_of_block = attr.attrib(
      type=typing.List[int], factory=list)
  # Skip residual connection is added to the input of this block
  # after a DenseLayer transformation (0-based index)
  skip_residue_connection_to_input_of_block = attr.attrib(
      type=typing.List[int], factory=list)
  # Dense layer prototype for skip residual connection of block inputs.
  residue_dense_layer = attr.attrib(
      type=typing.Optional[DenseLayer], default=None)
  # Initial scaling for each block output before internal residual
  # connection. Options: 'none', 'linear', 'reciprocal', 'exponential', 'zero'.
  scale_tdcn_block = attr.attrib(type=typing.Text, default='none')


def improved_tdcn(depth_multiplier=1):
  """Build ImprovedTDCN object for improved_tdcn."""
  normact = NormAndActivationLayer(
      norm_layer=NormLayer(
          norm_type='global_layer_norm', bin_wise=True),
      activation='prelu')
  netcfg = ImprovedTDCN(
      block_prototype_indices=([0] * (32 * depth_multiplier)),
      block_dilations=([1, 2, 4, 8, 16, 32, 64, 128] * (4 * depth_multiplier)),
      skip_residue_connection_from_input_of_block=[0, 0, 0, 8, 8, 16],
      skip_residue_connection_to_input_of_block=[8, 16, 24, 16, 24, 24],
      scale_tdcn_block='exponential')
  dense_biased = DenseLayer(scale=1.0, use_bias=True)
  netcfg.initial_dense_layer = copy.deepcopy(dense_biased)
  netcfg.residue_dense_layer = copy.deepcopy(dense_biased)
  netcfg.prototype_block = [TDCNBlock(
      bottleneck=256,
      num_conv_channels=512,
      kernel_size=3,
      dense1=DenseLayer(scale=1.0, activation='external'),
      dense2=DenseLayer(scale=-99, activation='external'),
      normact1=normact,
      normact2=normact,
      tclayer=TimeConvLayer(separable=True))]
  return netcfg
