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
"""A model to sequentially separate variable number of sources.

We use a sequential neural network using TDCN++ and beamforming.
We handle a variable number of sources up to a maximum number.

This model works both for single channel and multi-channel
mixture_waveforms input data.
"""

import typing
import attr

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from . import consistency
from . import groupwise
from . import multichannel_filtering
from . import network
from . import network_config
from . import shaper
from . import signal_transformer
from . import signal_util
from . import summaries
from . import summary_util

Shaper = shaper.Shaper

# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)


def _stabilized_log_base(x, base=10., stabilizer=1e-8):
  """Stabilized log with specified base."""
  logx = tf.log(x + stabilizer)
  logb = tf.log(tf.constant(base, dtype=logx.dtype))
  return logx / logb


def log_mse_loss(source, separated, max_snr=1e6, bias_ref_signal=None):
  """Negative log MSE loss, the negated log of SNR denominator."""
  err_pow = tf.reduce_sum(tf.square(source - separated), axis=-1)
  snrfactor = 10.**(-max_snr / 10.)
  if bias_ref_signal is None:
    ref_pow = tf.reduce_sum(tf.square(source), axis=-1)
  else:
    ref_pow = tf.reduce_sum(tf.square(bias_ref_signal), axis=-1)
  bias = snrfactor * ref_pow
  return 10. * _stabilized_log_base(bias + err_pow)


def _weights_for_nonzero_refs(source_waveforms, consider_as_zero=None):
  """Return shape (batch, source) weights for signals that are nonzero.

  Args:
    source_waveforms: A tensor (batch, source, samples), dtype=tf.float32.
    consider_as_zero: An optional tensor (batch, source), dtype=tf.bool which
      indicates some entries as being zero even if they are not exactly zero.
      This can be used to indicate some source types (e.g. noise) as being
      zero regardless of their norm.
  Returns:
    consider_nonzero: A tensor (batch, source), dtype=tf.bool of sources
      that will be considered as non-zero based on their norm being greater than
      1e-8 or they are not in consider_as_zero array if given.
  """
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  consider_nonzero = tf.greater(source_norms, 1e-8)
  if consider_as_zero is not None:
    consider_nonzero = tf.logical_and(consider_nonzero,
                                      tf.logical_not(consider_as_zero))
  return consider_nonzero


def _weights_for_signal_type(batch, signal_types, chosen_signal_types):
  """Return (batch, source) weights for signal types that are among chosen ones.

  Args:
    batch: Batch size dimension, integer.
    signal_types: A list of signal types, such as ['speech', 'speech', 'noise'].
    chosen_signal_types: A set that is a subset of set(signal_types) which
      includes source types that are going to be chosen in the output tensor.
  Returns:
    choose_source: A (batch, source) tensor of dtype=tf.bool which is 1 for
      sources that are among the chosen_signal_types, and 0 for others.
  """
  num_signals = len(signal_types)
  choose_source = [False] * num_signals
  if chosen_signal_types:
    for i, signal_type in enumerate(signal_types):
      if signal_type in chosen_signal_types:
        choose_source[i] = True
  choose_source = tf.convert_to_tensor(choose_source, dtype=tf.bool)
  choose_source = tf.reshape(choose_source, (1, num_signals))
  choose_source = tf.broadcast_to(choose_source, (batch, num_signals))
  return choose_source


def _weights_for_num_sources(source_waveforms, num_sources,
                             consider_as_zero=None):
  """Return shape (batch, source) weights for examples with num_sources."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  max_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
  consider_nonzero = tf.greater(source_norms, 1e-8)
  if consider_as_zero is not None:
    consider_nonzero = tf.logical_and(consider_nonzero,
                                      tf.logical_not(consider_as_zero))
  num_sources_per_example = tf.reduce_sum(
      tf.cast(consider_nonzero, tf.float32),
      axis=1, keepdims=True)
  has_num_sources = tf.equal(num_sources_per_example, num_sources)
  return tf.tile(has_num_sources, (1, max_sources))


@attr.attrs
class HParams(object):
  """Model hyperparameters."""
  # Weight on loss component for zero reference signals.
  loss_zero_ref_weight = attr.attrib(type=float, default=1.0)
  # mix_weights_type = Type of weights to use for mixture consistency. Options:
  #     ``: No mixture consistency.
  #     `uniform`: Uniform weight of 1 / num_sources (also called unweighted).
  #     `magsq`: Weight for source j is \sum_{mic, time} \hat{x}_j ^ 2
  #              over \sum_{source, mic, time} \hat{x}_j ^ 2.
  #     `pred_source`: Predict weights with shape (batch, source, 1).
  mix_weights_type = attr.attrib(type=typing.Text, default='pred_source')
  # List of signal names, e.g. ['signal_1', 'signal_2'].
  signal_names = attr.attrib(type=typing.List[typing.Text],
                             default=['background', 'foreground_1',
                                      'foreground_2', 'foreground_3'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  signal_types = attr.attrib(type=typing.List[typing.Text],
                             default=['source'] * 4)
  # If not None, signal_loss_weights per signal_type are specified through this
  # list the same length as signal_types. If None, all signal loss weights are
  # set to 1.0.
  signal_loss_weights = attr.attrib(
      type=typing.Optional[typing.List[float]], default=None)
  # List of signal types to consider as having a reference of zero for the
  # the purposes of loss calculations and summary reports.
  # It could be beneficial to add noise, or sensor_noise to this list.
  ref_zero_signal_types = attr.attrib(
      type=typing.Optional[typing.List[typing.Text]], default=None)
  # Sample rate of the input audio in hertz.
  sr = attr.attrib(type=float, default=16000.0)
  # Initial learning rate used by the optimizer.
  lr = attr.attrib(type=float, default=1e-4)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_steps = attr.attrib(type=int, default=2000000)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_rate = attr.attrib(type=float, default=0.5)
  # Gradient maximum norm for gradient clipping
  grads_max_norm = attr.attrib(type=float, default=5.0)
  # STFT window size in seconds.
  ws = attr.attrib(type=float, default=0.032)
  # STFT hop size in seconds.
  hs = attr.attrib(type=float, default=0.008)

  # TDCN++ network parameters.
  tdcn_bottleneck = attr.attrib(type=int, default=256)
  tdcn_conv_channels = attr.attrib(type=int, default=512)
  tdcn_kernel_size = attr.attrib(type=int, default=3)
  tdcn_num_dilations = attr.attrib(type=int, default=8)
  tdcn_num_repeats = attr.attrib(type=int, default=4)
  tdcn_norm_type = attr.attrib(type=typing.Text, default='instance_norm')
  tdcn_scale_type = attr.attrib(type=typing.Text, default='exponential')

  # Input feature normalization type. Use empty string '' for no normalization.
  # Options: 'mean_and_variance', 'compress_and_max', ''
  input_feature_norm_type = attr.attrib(type=typing.Text,
                                        default='mean_and_variance')

  # Number of iterations.
  iter_num = attr.attrib(type=int, default=1)
  # If True, apply loss to output of each iteration.
  iter_reploss = attr.attrib(type=bool, default=True)
  # If True, block gradient flow through iterations.
  iter_stopgrad = attr.attrib(type=bool, default=True)

  # Multimic beamforming parameters.
  ######################################################
  # How many mics to use for multichannel filtering.
  num_mics = attr.attrib(type=int, default=1)
  # Reference mic index.
  refmic = attr.attrib(type=int, default=0)
  # `;`-delimited list of which iterations to perform multichannel filtering.
  iters_do_mcf = attr.attrib(type=str, default='')
  # STFT window size for beamforming in seconds.
  mcf_ws = attr.attrib(type=float, default=0.064)
  # STFT hop size for beamforming in seconds.
  mcf_hs = attr.attrib(type=float, default=0.032)
  # Block size in seconds. Negative for full length.
  mcf_block_size_in_seconds = attr.attrib(type=float, default=-1.0)
  # Frame context length for multi-frame beamforming.
  mcf_frame_context_length = attr.attrib(type=int, default=1)
  # Frame context type for multi-frame beamforming.
  mcf_frame_context_type = attr.attrib(type=str, default='centered')
  # Beamformer type.
  mcf_beamformer_type = attr.attrib(type=str, default='wiener')


def get_model_hparams():
  return HParams()


def _get_input_feature(mixture_coeffs, norm_type=''):
  """Get input features of mask prediction network from mixture coefficients."""
  input_feature = tf.abs(mixture_coeffs)

  if norm_type == 'mean_and_variance':
    input_feature = network.LayerNormalizationScalarParams(
        axis=[-3, -2, -1],
        name='layer_norm_on_mag')(input_feature)
  elif norm_type == 'compress_and_max':
    input_feature = tf.pow(input_feature, 0.3)
    input_feature_max = 1e-3 + tf.reduce_max(input_feature, axis=[-3, -2, -1],
                                             keepdims=True)
    input_feature /= input_feature_max
    # Add learnable scale and bias after max normalization.
    input_feature = network.scale_layer(input_feature)
    input_feature = network.scalar_bias_layer(input_feature)
  elif not norm_type:
    # Add learnable scale and bias after taking magnitude coeffs directly.
    input_feature = network.scale_layer(input_feature)
    input_feature = network.scalar_bias_layer(input_feature)
  else:
    raise ValueError(f'Input feature norm type {norm_type} unknown.')
  return input_feature


def separate_waveforms(mixture_waveforms, hparams):
  """Computes and returns separated waveforms.

  Args:
    mixture_waveforms: Waveform of audio to separate, shape
      (batch, signals, time). First signal with index 0 is the mixture signal at
      a reference microphone and the other signals can be separated sources
      from a previous round.
    hparams: Model hyperparameters.
  Returns:
    Separated audio tensor, shape (batch, source, time), same type as mixture.
  """
  num_sources = len(hparams.signal_names)
  tensor_shaper = Shaper({'source': num_sources})

  # Compute encoder coefficients.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs,
      zeropad_beginning=True)
  mixture_coeffs = transformer.forward(mixture_waveforms)
  inverse_transform = transformer.inverse
  mixture_coeffs_input = _get_input_feature(mixture_coeffs,
                                            hparams.input_feature_norm_type)
  tensor_shaper.register_axes(mixture_coeffs, ['batch', 'sig', 'frame', 'bin'])
  mixture_coeffs_input = tensor_shaper.change(
      mixture_coeffs_input[:, :, tf.newaxis],
      ['batch', 'sig', 1, 'frame', 'bin'],
      ['batch', 'frame', 1, ('sig', 'bin')])

  # Obtain config and run the TDCN++ network.
  net_config = network_config.improved_tdcn(
      bottleneck=hparams.tdcn_bottleneck,
      conv_channels=hparams.tdcn_conv_channels,
      kernel_size=hparams.tdcn_kernel_size,
      num_dilations=hparams.tdcn_num_dilations,
      num_repeats=hparams.tdcn_num_repeats,
      norm_type=hparams.tdcn_norm_type,
      scale_type=hparams.tdcn_scale_type)
  core_activations = network.improved_tdcn(mixture_coeffs_input, net_config)
  tensor_shaper.register_axes(core_activations, ['batch', 'frame', 'out_depth'])

  # Apply a dense layer to increase output dimension.
  bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
  dense_config = network.update_config_from_kwargs(
      network_config.DenseLayer(),
      num_outputs=bins * num_sources,
      activation='linear')
  activations = network.dense_layer(core_activations, dense_config)
  tensor_shaper.register_axes(
      activations, ['batch', 'frame', ('source', 'bin')])

  # Create a mask from the output activations.
  activations = tensor_shaper.change(
      activations, ['batch', 'frame', ('source', 'bin')],
      ['batch', 'source', 'frame', 'bin'])
  mask = network.get_activation_fn('sigmoid')(activations)
  mask = tf.identity(mask, name='mask')

  # Apply the mask to the mixture coefficients.
  mask = tf.cast(mask, dtype=mixture_coeffs.dtype)
  # Use mixture signal as the mask input.
  mask_input = mixture_coeffs[:, :1]
  tensor_shaper.register_axes(mask_input, ['batch', 1, 'frame', 'bin'])
  separated_coeffs = mask * mask_input
  tensor_shaper.register_axes(
      separated_coeffs, ['batch', 'source', 'frame', 'bin'])

  # Reconstruct the separated waveforms from the masked coefficients.
  mixture_length = signal_util.static_or_dynamic_dim_size(mixture_waveforms, -1)
  separated_waveforms = inverse_transform(separated_coeffs)
  separated_waveforms = separated_waveforms[..., :mixture_length]

  # Apply mixture consistency, if specified.
  if hparams.mix_weights_type:
    if hparams.mix_weights_type == 'pred_source':
      # Mean-pool across time.
      mix_weights = tf.reduce_mean(core_activations, axis=1)
      # Dense layer to num_sources.
      dense_config = network.update_config_from_kwargs(
          network_config.DenseLayer(),
          num_outputs=num_sources,
          activation='linear')
      with tf.variable_scope('mix_weights'):
        mix_weights = network.dense_layer(mix_weights, dense_config)
      # Softmax across sources.
      mix_weights = tf.nn.softmax(
          mix_weights, axis=-1)[:, :, tf.newaxis, tf.newaxis]
      tensor_shaper.register_axes(
          mix_weights, ['batch', 'source', 1, 1])
    elif (hparams.mix_weights_type == 'uniform'
          or hparams.mix_weights_type == 'magsq'):
      mix_weights = None
    else:
      raise ValueError('Unknown mix_weights_type of "{}".'.format(
          hparams.mix_weights_type))
    # Expand separated_waveforms to source image shape
    # (batch, source, 1, samples) expected by the consistency projection code.
    separated_waveforms = tf.expand_dims(separated_waveforms, axis=2)
    separated_waveforms = consistency.enforce_mixture_consistency_time_domain(
        mixture_waveforms[:, :1], separated_waveforms,
        mix_weights=mix_weights,
        mix_weights_type=hparams.mix_weights_type)
    # Remove the added dimension.
    separated_waveforms = separated_waveforms[:, :, 0]

  return separated_waveforms


class LossMaker(object):
  """Class for calculating losses between reference and estimated signals."""

  def __init__(self, source_waveforms, signal_types,
               loss_zero_ref_weight=None, mixture_waveforms=None,
               max_snr=30, max_snr_for_zero_sources=20,
               loss_weights=None, ref_zero_signal_types=None):
    # Get (max) number of sources.
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    num_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
    self.signal_types = signal_types
    self.unique_signal_types = list(set(signal_types))
    self.max_snr = max_snr
    self.max_snr_for_zero_sources = max_snr_for_zero_sources
    self.source_waveforms = source_waveforms
    if not loss_weights:
      loss_weights = [1.0] * num_sources
    self.loss_weights = tf.reshape(
        tf.convert_to_tensor(loss_weights, tf.float32), (1, num_sources))
    self.loss_weights = tf.broadcast_to(self.loss_weights,
                                        (batch_size, num_sources))
    # Initial loss functions for resolving permutations.
    self.loss_fns = {signal_type: log_mse_loss for signal_type in
                     self.unique_signal_types}

    # Permutation-invariant training requires realigning the sources to match
    # the references, both in loss computations and computing summary metrics.

    # Build loss split between all-zero and nonzero reference signals.
    # Some signal types may be considered as always being zero for loss and
    # summary purposes.
    source_is_considered_zero = _weights_for_signal_type(
        batch_size, signal_types, ref_zero_signal_types)
    self.source_is_nonzero = _weights_for_nonzero_refs(
        source_waveforms, consider_as_zero=source_is_considered_zero)
    self.count_nonzero = tf.reduce_sum(
        tf.cast(self.source_is_nonzero, tf.float32))
    self.source_is_zero = tf.logical_not(self.source_is_nonzero)
    self.total_count = tf.cast(batch_size * num_sources, tf.float32)
    self.count_zero = self.total_count - self.count_nonzero

    # Waveforms with nonzero references.
    self.source_waveforms_nonzero = tf.boolean_mask(
        source_waveforms, self.source_is_nonzero)[:, tf.newaxis]

    # Since the sources are re-arranged when nonzero sources in the whole batch
    # are picked using a boolean_mask, we need to also pick and re-arrange
    # the loss weights corresponding to them.
    self.loss_weights_nonzero = tf.boolean_mask(
        self.loss_weights, self.source_is_nonzero)[:, tf.newaxis]

    self.source_waveforms_zero = tf.boolean_mask(
        source_waveforms, self.source_is_zero)[:, tf.newaxis]

    self.loss_weights_zero = tf.boolean_mask(
        self.loss_weights, self.source_is_zero)[:, tf.newaxis]

    self.loss_zero_ref_weight = loss_zero_ref_weight
    self.mixture_waveforms = mixture_waveforms
    # For loss for zero references.
    if self.loss_zero_ref_weight:
      self.mixture_waveforms_zero = tf.boolean_mask(
          tf.tile(self.mixture_waveforms, (1, num_sources, 1)),
          self.source_is_zero)[:, tf.newaxis]

  def add_loss(self, separated_waveforms):
    """Add loss for given separated_waveforms."""
    # Permute separated to match references through self.loss_fns.
    _, separated_waveforms = groupwise.apply(
        self.loss_fns, self.signal_types, self.source_waveforms,
        separated_waveforms, self.unique_signal_types)
    separated_waveforms_nonzero = tf.boolean_mask(
        separated_waveforms, self.source_is_nonzero)[:, tf.newaxis]
    separated_waveforms_zero = tf.boolean_mask(
        separated_waveforms, self.source_is_zero)[:, tf.newaxis]
    # Use eventual loss function as log_mse_loss.
    # Loss for zero references only if self.loss_zero_ref_weight is nonzero.
    if self.loss_zero_ref_weight:
      def loss_zero_fn():
        return tf.reduce_sum(self.loss_weights_zero * log_mse_loss(
            self.source_waveforms_zero,
            separated_waveforms_zero,
            max_snr=self.max_snr_for_zero_sources,
            bias_ref_signal=self.mixture_waveforms_zero))
      calc_losses_zero_ref = tf.logical_and(self.count_zero > 0,
                                            self.count_zero < self.total_count)
      loss_zero = tf.cond(calc_losses_zero_ref, loss_zero_fn, lambda: 0.0)
      loss_weight = self.loss_zero_ref_weight / self.total_count
      loss_zero = tf.identity(loss_weight * loss_zero, name='loss_ref_zero')
      tf.losses.add_loss(loss_zero)

    # Loss for nonzero references.
    def loss_nonzero_fn():
      return tf.reduce_sum(self.loss_weights_nonzero * log_mse_loss(
          self.source_waveforms_nonzero,
          separated_waveforms_nonzero,
          max_snr=self.max_snr))
    loss_nonzero = tf.cond(self.count_nonzero > 0.0, loss_nonzero_fn,
                           lambda: 0.0)
    loss_nonzero = tf.identity(loss_nonzero / self.total_count,
                               name='loss_ref_nonzero')
    tf.losses.add_loss(loss_nonzero)
    return separated_waveforms


def add_summaries_and_return_metrics(waveforms_to_summaries,
                                     mixture_waveforms,
                                     source_waveforms,
                                     source_is_zero,
                                     source_is_nonzero,
                                     hparams):
  """Adds summaries for all waveforms."""
  # Compute and define additional tensors to be used in summaries and metrics.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  source_spectrograms = transformer.forward(source_waveforms)
  mixture_spectrograms = transformer.forward(mixture_waveforms)
  scalars = {}
  weights = {}
  audio = {}
  images = {}
  for prefix, separated_waveforms in waveforms_to_summaries.items():
    new_scalars = {}
    new_weights = {}
    new_audio = {}
    new_images = {}
    separated_spectrograms = transformer.forward(separated_waveforms)
    if prefix:
      new_signal_names = [prefix + '_' + name for name in hparams.signal_names]
    else:
      new_signal_names = hparams.signal_names
    new_images.update(
        summaries.compute_spectrogram_summaries(
            new_signal_names,
            separated_spectrograms,
            source_spectrograms,
            mixture_spectrograms))
    new_audio.update(
        summaries.compute_audio_summaries(
            new_signal_names,
            separated_waveforms,
            source_waveforms,
            mixture_waveforms))
    new_scalars.update(
        summaries.scalar_snr_metrics(
            new_signal_names,
            separated_waveforms,
            source_waveforms,
            mixture_waveforms))
    # Separately report metrics for all-zero reference sources.
    scalars_zero, weights_zero = summaries.scalar_snr_metrics_weighted(
        new_signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        source_is_zero)
    new_scalars.update({name + '_ref_zero': value
                        for name, value in scalars_zero.items()})
    new_weights.update({name + '_ref_zero': value
                        for name, value in weights_zero.items()})
    # For 1src mixtures, get average metric for a nonzero reference source.
    # Compute shape (batch, source) boolean weights for examples with
    # 1 active signal.
    weights_for_1src = tf.logical_and(
        source_is_nonzero,  # Shape (batch, source) indicator of active srcs.
        _weights_for_num_sources(source_waveforms, 1,
                                 consider_as_zero=source_is_zero))
    # Compute scalars only for examples with 1 active signal.
    scalars_1src, weights_1src = summaries.scalar_snr_metrics_weighted(
        new_signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        weights_for_1src)
    new_scalars.update({name + '_1src_ref_nonzero': value
                        for name, value in scalars_1src.items()})
    new_weights.update({name + '_1src_ref_nonzero': value
                        for name, value in weights_1src.items()})
    # For min2srcs mixtures, average metric for nonzero reference sources.
    # Compute shape (batch, source) boolean weights for examples with
    # 2+ active signals.
    weights_for_min2srcs = tf.logical_and(
        source_is_nonzero,  # Shape (batch, source) indicator of active srcs.
        tf.logical_not(_weights_for_num_sources(
            source_waveforms, 1, consider_as_zero=source_is_zero)))
    # Compute scalars only for examples with 2+ active signals.
    scalars_min2srcs, weights_min2srcs = summaries.scalar_snr_metrics_weighted(
        new_signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        weights_for_min2srcs)
    new_scalars.update({name + '_min2srcs_ref_nonzero': value
                        for name, value in scalars_min2srcs.items()})
    new_weights.update({name + '_min2srcs_ref_nonzero': value
                        for name, value in weights_min2srcs.items()})
    # Unless prefix is '' (final separated), exclude duplicate mixture
    # summaries. Also, fix the _signals_ summaries for non-empty prefix.
    if prefix:
      new_audio = {
          k: v for k, v in new_audio.items() if 'mixture_' not in k
          and 'source_' not in k}
      new_images = {
          k: v for k, v in new_images.items() if 'mixture_' not in k
          and 'source_' not in k}
      new_scalars = {
          k: v for k, v in new_scalars.items() if 'mixture_' not in k}
      new_scalars = {k.replace('_signals_', f'_{prefix}_signals_'): v
                     for k, v in new_scalars.items()}
      new_weights = {
          k: v for k, v in new_weights.items() if 'mixture_' not in k}
      new_weights = {k.replace('_signals_', f'_{prefix}_signals_'): v
                     for k, v in new_weights.items()}
    scalars.update(new_scalars)
    weights.update(new_weights)
    audio.update(new_audio)
    images.update(new_images)

  tf_losses = tf.losses.get_losses()
  loss_names = [_.name for _ in tf_losses]
  loss_tensor = tf.expand_dims(tf.stack(tf_losses), axis=0)
  scalars.update(summaries.compute_loss_summaries(loss_names, loss_tensor))

  # metrics dict is used in evaluation and we need to include per-batch
  # weights in them (if any) to calculate overall metrics correctly.
  metrics = {name: tf.metrics.mean(s, weights=weights.get(name, None))
             for name, s in scalars.items()}
  # We do not need per-batch weights in train summaries since train summaries
  # are over a single batch.
  summary_util.create_summaries(sample_rate=hparams.sr, scalars=scalars,
                                audio=audio, images=images)
  return metrics


def model_fn(features, labels, mode, params):
  """Constructs a sequential/iterative source separation model with summaries.

  Args:
    features: Dictionary {name: Tensor} of model inputs.
    labels: Any training-only inputs.
    mode: Build mode, one of tf.estimator.ModeKeys.
    params: Dictionary of Model hyperparameters.

  Returns:
    EstimatorSpec describing the model.
  """
  del labels

  hparams = params['hparams']

  # Only use specified number of mics.
  if hparams.num_mics:
    features['receiver_audio'] = (
        features['receiver_audio'][:, :hparams.num_mics])
    if 'source_images' in features:
      features['source_images'] = (
          features['source_images'][:, :, :hparams.num_mics])

  mixture_waveforms = features['receiver_audio']

  refmic = hparams.refmic
  mixture_waveforms_refmic = mixture_waveforms[:, refmic:refmic+1]
  if mode != tf_estimator.ModeKeys.PREDICT:
    # Get reference sources at reference mic.
    source_waveforms = features['source_images'][:, :, refmic]
    loss_maker = LossMaker(source_waveforms, hparams.signal_types,
                           hparams.loss_zero_ref_weight,
                           mixture_waveforms_refmic,
                           loss_weights=hparams.signal_loss_weights,
                           ref_zero_signal_types=hparams.ref_zero_signal_types)

  separated_waveforms = None
  # To avoid too many summaries, we only include the first iteration separated,
  # plus final separated waveform which
  # has scope_prefix = '' and the last beamformed waveform only.
  prefixes_in_summaries = ['', 'model_iter_0']
  last_bf_iter = hparams.iters_do_mcf.split(';')[-1]
  if last_bf_iter:
    prefixes_in_summaries.append(f'model_iter_{last_bf_iter}_beamformed')
  waveforms_to_summaries = {}
  for i in range(hparams.iter_num):
    first_iter = i == 0
    last_iter = i == (hparams.iter_num - 1)
    scope_prefix = 'model_iter_%d' % i if not last_iter else ''
    scope = tf.get_variable_scope().name
    scope = scope_prefix + '/' + scope if scope_prefix else scope
    with tf.variable_scope(scope):

      # Build the input for this iteration.
      if first_iter:
        input_waveforms = mixture_waveforms_refmic
        # input_waveforms is shape (batch, 1, time).
      else:
        # Concat (batch, 1, time) with (batch, source, time) along axis 1.
        input_waveforms = tf.concat(
            [mixture_waveforms_refmic, separated_waveforms], 1)
        # input_waveforms is (batch, 1 + source, time).

      # Separate waveforms for this iteration.
      separated_waveforms = separate_waveforms(input_waveforms, hparams)
      if scope_prefix in prefixes_in_summaries:
        waveforms_to_summaries[scope_prefix] = separated_waveforms

      if mode != tf_estimator.ModeKeys.PREDICT and not last_iter:

        if hparams.iter_reploss:
          # Add loss on this iteration's separated waveforms.
          loss_maker.add_loss(separated_waveforms)

        if hparams.iter_stopgrad:
          # Stop gradients between iterations.
          separated_waveforms = tf.stop_gradient(separated_waveforms)

      # Perform multichannel filtering.
      if hparams.iters_do_mcf:
        iters_do_mcf = [int(s) for s in hparams.iters_do_mcf.split(';')]
      else:
        iters_do_mcf = []
      if i in iters_do_mcf:
        beamformed = (
            multichannel_filtering.compute_multichannel_filter_from_signals(
                mixture_waveforms, separated_waveforms, refmic=hparams.refmic,
                sample_rate=hparams.sr, ws=hparams.mcf_ws, hs=hparams.mcf_hs,
                frame_context_length=hparams.mcf_frame_context_length,
                frame_context_type=hparams.mcf_frame_context_type,
                block_size_in_seconds=hparams.mcf_block_size_in_seconds,
                beamformer_type=hparams.mcf_beamformer_type,
                ))
        beamformed = tf.identity(beamformed, 'beamformed_waveforms')
        scope_prefix_bf = scope_prefix + '_beamformed'
        if scope_prefix_bf in prefixes_in_summaries:
          waveforms_to_summaries[scope_prefix_bf] = beamformed
        separated_waveforms = beamformed

  predictions = {'separated_waveforms': separated_waveforms}

  separated_waveforms = tf.identity(separated_waveforms,
                                    name='denoised_waveforms')
  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Add loss function on final separated_waveforms.
  loss_maker.add_loss(separated_waveforms)

  # Build the optimizer.
  loss = tf.losses.get_total_loss()
  learning_rate = tf.train.exponential_decay(
      hparams.lr,
      tf.train.get_or_create_global_step(),
      decay_steps=hparams.lr_decay_steps,
      decay_rate=hparams.lr_decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  if params.get('use_tpu', False):
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  # Build the train_op.
  gradients, variables = zip(*optimizer.compute_gradients(loss))
  if hparams.grads_max_norm:
    gradients, _ = tf.clip_by_global_norm(gradients, hparams.grads_max_norm)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(zip(gradients, variables),
                                       global_step=global_step)

  # Resolve permutations of all waveforms to send to summaries.
  permuted_waveforms_to_summaries = {}
  for prefix, separated_waveforms in waveforms_to_summaries.items():
    _, separated_waveforms = groupwise.apply(
        loss_maker.loss_fns, loss_maker.signal_types, source_waveforms,
        separated_waveforms, loss_maker.unique_signal_types)
    permuted_waveforms_to_summaries[prefix] = separated_waveforms

  metrics = add_summaries_and_return_metrics(
      permuted_waveforms_to_summaries, mixture_waveforms_refmic,
      source_waveforms, loss_maker.source_is_zero,
      loss_maker.source_is_nonzero, hparams)

  logging_hook = tf.train.LoggingTensorHook({'loss': loss}, every_n_secs=10)

  return tf_estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=metrics,
      train_op=train_op,
      training_hooks=[logging_hook])

