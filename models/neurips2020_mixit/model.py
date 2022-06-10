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
"""A model to separate waveforms using TDCN++ with MixIT."""

import functools
import os
import sys
import typing

import attr
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

cur_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(cur_path))
train_module_path = os.path.join(parent_path, 'models')
sys.path.append(train_module_path)
from train import consistency
from train import groupwise
from train import mixit
from train import network
from train import network_config
from train import shaper
from train import signal_transformer
from train import signal_util
from train import summaries
from train import summary_util
Shaper = shaper.Shaper

# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)


def _stabilized_log_base(x, base=10., stabilizer=1e-8):
  """Stabilized log with specified base."""
  logx = tf.log(x + stabilizer)
  logb = tf.log(tf.constant(base, dtype=logx.dtype))
  return logx / logb


def log_mse_loss(source, separated, max_snr=30.0, bias_ref_signal=None):
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
  # mix_weights_type = Type of weights to use for mixture consistency. Options:
  #     ``: No mixture consistency.
  #     `uniform`: Uniform weight of 1 / num_sources (also called unweighted).
  #     `magsq`: Weight for source j is \sum_{mic, time} \hat{x}_j ^ 2
  #              over \sum_{source, mic, time} \hat{x}_j ^ 2.
  #     `pred_source`: Predict weights with shape (batch, source, 1).
  mix_weights_type = attr.attrib(type=typing.Text, default='pred_source')
  # List of signal names, e.g. ['signal_1', 'signal_2'].
  signal_names = attr.attrib(type=typing.List[typing.Text],
                             default=['mix1_background', 'mix1_foreground_1',
                                      'mix1_foreground_2', 'mix1_foreground_3',
                                      'mix2_background', 'mix2_foreground_1',
                                      'mix2_foreground_2', 'mix2_foreground_3'])
  # A list of strings same length as signal_names specifying signal type, used
  # for groupwise permutation-invariance.
  signal_types = attr.attrib(type=typing.List[typing.Text],
                             default=['source'] * 8)
  # Sample rate of the input audio in hertz.
  sr = attr.attrib(type=float, default=16000.0)
  # Initial learning rate used by the optimizer.
  lr = attr.attrib(type=float, default=1e-4)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_steps = attr.attrib(type=int, default=2000000)
  # Decay lr by lr_decay_rate every lr_step_steps.
  lr_decay_rate = attr.attrib(type=float, default=0.5)
  # Use learnable basis (True) or STFT (False).
  learn_basis = attr.attrib(type=bool, default=True)
  # Number of basis coefficients. Unused if learn_basis is False.
  num_coeffs = attr.attrib(type=int, default=256)
  # Basis window size in seconds.
  ws = attr.attrib(type=float, default=0.0025)
  # Basis hop size in seconds.
  hs = attr.attrib(type=float, default=0.00125)


def get_model_hparams():
  return HParams()


def conv_encoder(input_waveforms, samples_per_window, samples_per_hop,
                 num_coeffs):
  """Creates a convolutional encoder, as in the Conv-TasNet paper [1].

  [1] Yi Luo, Nima Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency
  Magnitude Masking for Speech Separation" https://arxiv.org/pdf/1809.07454.pdf.

  Args:
    input_waveforms: Tensor (..., input_samples)
    samples_per_window: int, how many samples per window.
    samples_per_hop: int, how many samples per hop.
    num_coeffs: int, how many basis coefficients.

  Returns:
    encoder_coeffs: coefficients of the encoder of shape
        (..., num_frames, num_coeffs), where num_frames is
        ceil[ (input_samples - samples_per_window) / samples_per_hop ] + 1.
  """

  with tf.variable_scope('conv_encoder'):
    input_frames = tf.signal.frame(input_waveforms,
                                   samples_per_window,
                                   samples_per_hop,
                                   pad_end=True)
    input_samples = signal_util.static_or_dynamic_dim_size(input_waveforms, -1)
    expected_num_frames = 1 - (-(input_samples - samples_per_window)
                               // samples_per_hop)
    input_frames = input_frames[..., :expected_num_frames, :]

    encoder_coeffs = tf.keras.layers.Dense(
        num_coeffs, activation='relu', use_bias=False).apply(input_frames)

  return encoder_coeffs


def conv_decoder(input_coeffs, samples_per_window, samples_per_hop):
  """Creates a convolutional decoder, as in the Conv-TasNet paper [1].

  [1] Yi Luo, Nima Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency
  Magnitude Masking for Speech Separation" https://arxiv.org/pdf/1809.07454.pdf.

  Args:
    input_coeffs: Tensor (batch_size, ..., num_frames, num_coeffs) of
      nonnegative coefficients, of dtype float32.
    samples_per_window: int, how many samples per window.
    samples_per_hop: int, how many samples per hop.

  Returns:
    waveforms: reconstructed time-domain signals of shape
        (batch_size, ..., num_coeffs + (num_frames-1)*frame_hop).
  """

  shape_orig = input_coeffs.shape.as_list()[:-2]
  batch_size = signal_util.static_or_dynamic_dim_size(input_coeffs, 0)
  num_frames = signal_util.static_or_dynamic_dim_size(input_coeffs, -2)
  num_coeffs = signal_util.static_or_dynamic_dim_size(input_coeffs, -1)

  if len(shape_orig) > 2:
    input_coeffs = tf.reshape(input_coeffs,
                              (batch_size, -1, num_frames, num_coeffs))
  # input_coeffs is of shape (batch_size, depth, num_frames, num_coeffs).

  with tf.variable_scope('conv_decoder'):

    # Multiply the decoder basis of shape (num_coeffs, samples_per_window) with
    # each frame.
    reconstructed_frames = tf.keras.layers.Dense(
        samples_per_window, activation=None, use_bias=False).apply(input_coeffs)
    # reconstructed_frames is shape
    # (batch_size, depth, num_frames, samples_per_window).

    # Overlap-add the frames to produce reconstructed waveforms.
    waveforms = tf.signal.overlap_and_add(reconstructed_frames, samples_per_hop)

    if len(shape_orig) > 2:
      output_samples = signal_util.static_or_dynamic_dim_size(waveforms, -1)
      waveforms = tf.reshape(waveforms, shape_orig + [output_samples])

    return waveforms


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
  if hparams.learn_basis:
    # Use learnable basis.
    samples_per_window = int(round(hparams.ws * hparams.sr))
    samples_per_hop = int(round(hparams.hs * hparams.sr))
    mixture_coeffs = conv_encoder(mixture_waveforms, samples_per_window,
                                  samples_per_hop, hparams.num_coeffs)
    inverse_transform = functools.partial(conv_decoder,
                                          samples_per_window=samples_per_window,
                                          samples_per_hop=samples_per_hop)
    mixture_coeffs_input = mixture_coeffs

  else:
    # Use STFT basis.
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

  batch_size = signal_util.static_or_dynamic_dim_size(mixture_waveforms, 0)

  # Create mixtures of mixtures (MoMs) on-the-fly by splitting batch in half.
  if mode == tf_estimator.ModeKeys.TRAIN or mode == tf_estimator.ModeKeys.EVAL:
    mixture_waveforms_1mix = mixture_waveforms
    # Build MoMs by splitting batch in half.
    with tf.control_dependencies([tf.compat.v1.assert_equal(
        tf.mod(batch_size, 2), 0)]):
      mixture_waveforms = tf.reshape(mixture_waveforms,
                                     (batch_size // 2, 2, -1))

    # Create the MoMs by summing up single mixtures.
    mix_of_mix_waveforms = tf.reduce_sum(mixture_waveforms, axis=1,
                                         keepdims=True)

  else:
    # Inference mode, mixture_waveforms is just an input placeholder.
    mix_of_mix_waveforms = mixture_waveforms

  # In eval mode, separate both MoMs and single mixtures.
  if mode == tf_estimator.ModeKeys.EVAL:
    input_waveforms = tf.concat([mix_of_mix_waveforms,
                                 mixture_waveforms_1mix], axis=0)
  else:
    input_waveforms = mix_of_mix_waveforms

  # Separate the input waveforms.
  separated_waveforms = separate_waveforms(input_waveforms, hparams)

  # In eval mode, split into separated from MoMs and from single mixtures.
  if mode == tf_estimator.ModeKeys.EVAL:
    # Separated sources from single mixtures.
    separated_waveforms_1mix = separated_waveforms[batch_size // 2:, :, :]
    # Separated sources from MoMs.
    separated_waveforms = separated_waveforms[:batch_size // 2, :, :]

  predictions = {'separated_waveforms': separated_waveforms}
  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Get reference sources.
  source_waveforms = features['source_images'][:, :, 0]
  max_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
  source_waveforms_1mix = tf.concat([source_waveforms,
                                     tf.zeros_like(source_waveforms)], axis=1)
  if batch_size > 1:
    source_waveforms = tf.reshape(source_waveforms,
                                  (batch_size // 2, 2 * max_sources, -1))
  else:
    source_waveforms = tf.concat([source_waveforms,
                                  tf.zeros_like(source_waveforms)], axis=1)

  # MixIT loss.
  loss, _ = mixit.apply(log_mse_loss, mixture_waveforms, separated_waveforms)
  loss = tf.identity(tf.reduce_mean(loss), name='loss_mixit')
  tf.losses.add_loss(loss)

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

  # Permute separated to match references for summaries.
  unique_signal_types = list(set(hparams.signal_types))
  loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}
  _, separated_waveforms = groupwise.apply(
      loss_fns, hparams.signal_types, source_waveforms, separated_waveforms,
      unique_signal_types)
  if mode == tf_estimator.ModeKeys.EVAL:
    # Also align sources separated from single mixtures.
    _, separated_waveforms_1mix = groupwise.apply(
        loss_fns, hparams.signal_types, source_waveforms_1mix,
        separated_waveforms_1mix, unique_signal_types)

  # In eval mode, evaluate separated from single mixtures, instead of from MoMs.
  if mode == tf_estimator.ModeKeys.EVAL:
    separated_waveforms = separated_waveforms_1mix
    source_waveforms = source_waveforms_1mix
    mix_of_mix_waveforms = mixture_waveforms_1mix

  # Compute spectrograms to be used in summaries.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=hparams.sr,
      window_time_seconds=hparams.ws,
      hop_time_seconds=hparams.hs)
  source_spectrograms = transformer.forward(source_waveforms)
  mixture_spectrograms = transformer.forward(mix_of_mix_waveforms)
  separated_spectrograms = transformer.forward(separated_waveforms)

  summary_dict = {}

  # Audio summaries.
  summary_dict['audio'] = summaries.compute_audio_summaries(
      signal_names=hparams.signal_names,
      separated_waveforms=separated_waveforms,
      source_waveforms=source_waveforms,
      mixture_waveforms=mix_of_mix_waveforms)

  # Spectrogram image summaries.
  summary_dict['images'] = summaries.compute_spectrogram_summaries(
      signal_names=hparams.signal_names,
      separated_spectrograms=separated_spectrograms,
      source_spectrograms=source_spectrograms,
      mixture_spectrograms=mixture_spectrograms)

  scalars = {}
  weights = {}
  # Only compute scalar summaries for nonzero reference sources.
  source_is_nonzero = _weights_for_nonzero_refs(source_waveforms)

  # Metrics for single-source examples.
  weights_1src = tf.logical_and(
      source_is_nonzero,
      _weights_for_num_sources(source_waveforms, 1))
  scalars_1src, weights_1src = summaries.scalar_snr_metrics_weighted(
      hparams.signal_names,
      separated_waveforms,
      source_waveforms,
      mix_of_mix_waveforms,
      weights_1src)
  scalars.update({name + '_1src_ref_nonzero': value
                  for name, value in scalars_1src.items()})
  weights.update({name + '_1src_ref_nonzero': value
                  for name, value in weights_1src.items()})

  # Metrics for multi-source examples.
  max_sources = len(hparams.signal_names)
  if max_sources > 1:
    weights_multisource = _weights_for_num_sources(source_waveforms, 2)
    for num_sources in range(3, max_sources + 1):
      weights_multisource = tf.logical_or(
          weights_multisource,
          _weights_for_num_sources(source_waveforms, num_sources))
    weights_multisource = tf.logical_and(source_is_nonzero, weights_multisource)
    scalars_msrc, weights_msrc = summaries.scalar_snr_metrics_weighted(
        hparams.signal_names,
        separated_waveforms,
        source_waveforms,
        mix_of_mix_waveforms,
        weights_multisource)
    scalars.update({name + '_min2src_ref_nonzero': value
                    for name, value in scalars_msrc.items()})
    weights.update({name + '_min2src_ref_nonzero': value
                    for name, value in weights_msrc.items()})

  summary_dict['scalars'] = scalars
  summary_util.create_summaries(sample_rate=hparams.sr, **summary_dict)
  metrics = {name: tf.metrics.mean(s, weights=weights.get(name, None))
             for name, s in scalars.items()}

  logging_hook = tf.train.LoggingTensorHook({'loss': loss}, every_n_secs=10)

  return tf_estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=metrics,
      train_op=train_op,
      training_hooks=[logging_hook])
