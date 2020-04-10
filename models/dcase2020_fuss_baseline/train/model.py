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
"""A model to separate waveforms using TDCN++."""

import attr
import typing

import numpy as np
import tensorflow.compat.v1 as tf

from . import consistency
from . import groupwise
from . import network
from . import network_config
from . import signal_transformer
from . import signal_util
from . import summaries
from . import summary_util
from . import shaper
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


def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (batch, source) weights for signals that are nonzero."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  return tf.greater(source_norms, 1e-8)


def _weights_for_num_sources(source_waveforms, num_sources):
  """Return shape (batch, source) weights for examples with num_sources."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  max_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
  num_sources_per_example = tf.reduce_sum(
      tf.cast(tf.greater(source_norms, 1e-8), tf.float32),
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
  # Sample rate of the input audio in hertz.
  sr = attr.attrib(type=float, default=16000.0)
  # Initial learning rate used by the optimizer.
  lr = attr.attrib(type=float, default=1e-4)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_steps = attr.attrib(type=int, default=2000000)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_rate = attr.attrib(type=float, default=0.5)
  # STFT window size in seconds.
  ws = attr.attrib(type=float, default=0.032)
  # STFT hop size in seconds.
  hs = attr.attrib(type=float, default=0.008)


def get_model_hparams():
  return HParams()


def separate_waveforms(mixture_waveforms, hparams):
  """Computes and returns separated waveforms.

  Args:
    mixture_waveforms: Waveform of audio to separate, shape (batch, mic, time).
    hparams: Model hyperparameters.
  Returns:
    Separated audio tensor, shape (batch, source, time), same type as mixture.
  """
  num_sources = len(hparams.signal_names)
  num_mics = signal_util.static_or_dynamic_dim_size(mixture_waveforms, 1)
  shaper = Shaper({'source': num_sources, '1': 1})

  # Compute encoder coefficients.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  mixture_coeffs = transformer.forward(mixture_waveforms)
  inverse_transform = transformer.inverse
  mixture_coeffs_input = tf.abs(mixture_coeffs)
  mixture_coeffs_input = network.LayerNormalizationScalarParams(
      axis=[-3, -2, -1],
      name='layer_norm_on_mag').apply(mixture_coeffs_input)
  shaper.register_axes(mixture_coeffs, ['batch', 'mic', 'frame', 'bin'])
  mixture_coeffs_input = shaper.change(mixture_coeffs_input[:, :, tf.newaxis],
                                       ['batch', 'mic', '1', 'frame', 'bin'],
                                       ['batch', 'frame', '1', ('mic', 'bin')])

  # Run the TDCN++ network.
  net_config = network_config.improved_tdcn()
  core_activations = network.improved_tdcn(mixture_coeffs_input, net_config)
  shaper.register_axes(core_activations, ['batch', 'frame', 'out_depth'])

  # Apply a dense layer to increase output dimension.
  bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
  dense_config = network.update_config_from_kwargs(
      network_config.DenseLayer(),
      num_outputs=num_mics * bins * num_sources,
      activation='linear')
  activations = network.dense_layer(core_activations, dense_config)
  shaper.register_axes(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')])

  # Create a mask from the output activations.
  activations = shaper.change(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')],
      ['batch', 'source', 'mic', 'frame', 'bin'])
  mask = network.get_activation_fn('sigmoid')(activations)
  mask = tf.identity(mask, name='mask')

  # Apply the mask to the mixture coefficients.
  mask = tf.cast(mask, dtype=mixture_coeffs.dtype)
  mask_input = mixture_coeffs[:, tf.newaxis]
  shaper.register_axes(mask_input, ['batch', '1', 'mic', 'frame', 'bin'])
  separated_coeffs = mask * mask_input
  shaper.register_axes(
      separated_coeffs, ['batch', 'source', 'mic', 'frame', 'bin'])

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
      shaper.register_axes(
          mix_weights, ['batch', 'source', '1', '1'])
    elif (hparams.mix_weights_type == 'uniform'
          or hparams.mix_weights_type == 'magsq'):
      mix_weights = None
    else:
      raise ValueError('Unknown mix_weights_type of "{}".'.format(
          hparams.mix_weights_type))
    separated_waveforms = consistency.enforce_mixture_consistency_time_domain(
        mixture_waveforms, separated_waveforms,
        mix_weights=mix_weights,
        mix_weights_type=hparams.mix_weights_type)

  # If multi-mic, just use the reference microphone.
  separated_waveforms = separated_waveforms[:, :, 0]

  separated_waveforms = tf.identity(separated_waveforms,
                                    name='denoised_waveforms')
  return separated_waveforms


def model_fn(features, labels, mode, params):
  """Constructs a spectrogram_lstm model with summaries.

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

  mixture_waveforms = features['receiver_audio']

  separated_waveforms = separate_waveforms(mixture_waveforms, hparams)

  predictions = {'separated_waveforms': separated_waveforms}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)

  # Get reference sources.
  source_waveforms = features['source_images'][:, :, 0]
  batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)

  # Permute separated to match references.
  unique_signal_types = list(set(hparams.signal_types))
  loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}
  _, separated_waveforms = groupwise.apply(
      loss_fns, hparams.signal_types, source_waveforms, separated_waveforms,
      unique_signal_types)
  # Permutation-invariant training requires realigning the sources to match
  # the references, both in loss computations and computing summary metrics.

  # Build loss split between all-zero and nonzero reference signals.
  source_is_nonzero = _weights_for_nonzero_refs(source_waveforms)
  source_is_zero = tf.logical_not(source_is_nonzero)

  # Get batch size and (max) number of sources.
  num_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)

  # Waveforms with nonzero references.
  source_waveforms_nonzero = tf.boolean_mask(
      source_waveforms, source_is_nonzero)[:, tf.newaxis]
  separated_waveforms_nonzero = tf.boolean_mask(
      separated_waveforms, source_is_nonzero)[:, tf.newaxis]

  # Waveforms with all-zero references.
  source_waveforms_zero = tf.boolean_mask(
      source_waveforms, source_is_zero)[:, tf.newaxis]
  separated_waveforms_zero = tf.boolean_mask(
      separated_waveforms, source_is_zero)[:, tf.newaxis]

  weight = 1. / tf.cast(batch_size * num_sources, tf.float32)

  # Loss for zero references.
  if hparams.loss_zero_ref_weight:
    mixture_waveforms_zero = tf.boolean_mask(
        tf.tile(mixture_waveforms[:, 0:1], (1, num_sources, 1)),
        source_is_zero)[:, tf.newaxis]
    loss = tf.reduce_sum(log_mse_loss(source_waveforms_zero,
                                      separated_waveforms_zero,
                                      max_snr=20,
                                      bias_ref_signal=mixture_waveforms_zero))
    loss_zero = tf.identity(hparams.loss_zero_ref_weight * weight * loss,
                            name='loss_ref_zero')
    tf.losses.add_loss(loss_zero)

  # Loss for nonzero references.
  loss = tf.reduce_sum(log_mse_loss(source_waveforms_nonzero,
                                    separated_waveforms_nonzero,
                                    max_snr=30))
  loss_nonzero = tf.identity(weight * loss, name='loss_ref_nonzero')
  tf.losses.add_loss(loss_nonzero)

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
  train_op = optimizer.minimize(
      loss,
      global_step=tf.compat.v1.train.get_or_create_global_step())

  # Compute and define additional tensors to be used in summaries and metrics.
  source_spectrograms = transformer.forward(source_waveforms)
  mixture_spectrograms = transformer.forward(mixture_waveforms)
  separated_spectrograms = transformer.forward(separated_waveforms)

  summary_dict = summaries.compute_summaries(
      signal_names=hparams.signal_names,
      separated_spectrograms=separated_spectrograms,
      source_spectrograms=source_spectrograms,
      mixture_spectrograms=mixture_spectrograms,
      separated_waveforms=separated_waveforms,
      source_waveforms=source_waveforms,
      mixture_waveforms=mixture_waveforms)
  summary_util.create_summaries(sample_rate=hparams.sr, **summary_dict)
  scalars = summary_dict['scalars']

  # Metrics only for nonzero reference sources.
  weights = {}
  scalars_nonzero, weights_nonzero = summaries.scalar_snr_metrics_weighted(
      hparams.signal_names,
      separated_waveforms,
      source_waveforms,
      mixture_waveforms,
      source_is_nonzero)
  scalars.update({name + '_ref_nonzero': value
                  for name, value in scalars_nonzero.items()})
  weights.update({name + '_ref_nonzero': value
                  for name, value in weights_nonzero.items()})
  # Metrics only for all-zero reference sources.
  scalars_zero, weights_zero = summaries.scalar_snr_metrics_weighted(
      hparams.signal_names,
      separated_waveforms,
      source_waveforms,
      mixture_waveforms,
      source_is_zero)
  scalars.update({name + '_ref_zero': value
                  for name, value in scalars_zero.items()})
  weights.update({name + '_ref_zero': value
                  for name, value in weights_zero.items()})
  max_sources = len(hparams.signal_names)
  num_sources_for_summaries = range(1, max_sources + 1)
  # For each number of sources, metrics only for those examples.
  for num_sources in num_sources_for_summaries:
    # Compute shape (batch, source) boolean weights for examples with
    # num_sources active signals.
    weights_for_nsrc = tf.logical_and(
        source_is_nonzero,  # Shape (batch, source) indicator of active signals.
        _weights_for_num_sources(source_waveforms, num_sources))
    # Compute scalar metrics only for examples with num_sources active signals.
    scalars_nsrc, weights_nsrc = summaries.scalar_snr_metrics_weighted(
        hparams.signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        weights_for_nsrc)
    scalars.update({name + '_%dsrcs_ref_nonzero' % num_sources: value
                    for name, value in scalars_nsrc.items()})
    weights.update({name + '_%dzero_ref_nonzero' % num_sources: value
                    for name, value in weights_nsrc.items()})
  metrics = {name: tf.metrics.mean(s, weights=weights.get(name, None))
             for name, s in scalars.items()}

  logging_hook = tf.train.LoggingTensorHook({'loss': loss}, every_n_secs=10)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=metrics,
      train_op=train_op,
      training_hooks=[logging_hook])
