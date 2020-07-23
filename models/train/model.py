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

import typing
import attr

import tensorflow.compat.v1 as tf

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
  tensor_shaper = Shaper({'source': num_sources, '1': 1})

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
  tensor_shaper.register_axes(mixture_coeffs, ['batch', 'mic', 'frame', 'bin'])
  mixture_coeffs_input = tensor_shaper.change(
      mixture_coeffs_input[:, :, tf.newaxis],
      ['batch', 'mic', '1', 'frame', 'bin'],
      ['batch', 'frame', '1', ('mic', 'bin')])

  # Run the TDCN++ network.
  net_config = network_config.improved_tdcn()
  core_activations = network.improved_tdcn(mixture_coeffs_input, net_config)
  tensor_shaper.register_axes(core_activations, ['batch', 'frame', 'out_depth'])

  # Apply a dense layer to increase output dimension.
  bins = signal_util.static_or_dynamic_dim_size(mixture_coeffs, -1)
  dense_config = network.update_config_from_kwargs(
      network_config.DenseLayer(),
      num_outputs=num_mics * bins * num_sources,
      activation='linear')
  activations = network.dense_layer(core_activations, dense_config)
  tensor_shaper.register_axes(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')])

  # Create a mask from the output activations.
  activations = tensor_shaper.change(
      activations, ['batch', 'frame', ('mic', 'source', 'bin')],
      ['batch', 'source', 'mic', 'frame', 'bin'])
  mask = network.get_activation_fn('sigmoid')(activations)
  mask = tf.identity(mask, name='mask')

  # Apply the mask to the mixture coefficients.
  mask = tf.cast(mask, dtype=mixture_coeffs.dtype)
  mask_input = mixture_coeffs[:, tf.newaxis]
  tensor_shaper.register_axes(mask_input, ['batch', '1', 'mic', 'frame', 'bin'])
  separated_coeffs = mask * mask_input
  tensor_shaper.register_axes(
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
      tensor_shaper.register_axes(
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

  # If multi-mic output, just use the reference microphone.
  if tensor_shaper.axis_sizes['mic'] > 1:
    separated_waveforms = separated_waveforms[:, :, hparams.refmic]
  else:
    separated_waveforms = separated_waveforms[:, :, 0]

  separated_waveforms = tf.identity(separated_waveforms,
                                    name='denoised_waveforms')
  return separated_waveforms


class LossMaker(object):
  """Class for losses between reference and estimated signals."""

  def __init__(self, source_waveforms, signal_types,
               loss_zero_ref_weight=None, mixture_waveforms=None,
               max_snr=30, max_snr_for_zero_sources=20):
    # Get batch size and (max) number of sources.
    batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
    num_sources = signal_util.static_or_dynamic_dim_size(source_waveforms, 1)
    self.signal_types = signal_types
    self.unique_signal_types = list(set(signal_types))
    self.max_snr = max_snr
    self.max_snr_for_zero_sources = max_snr_for_zero_sources
    self.source_waveforms = source_waveforms
    self.loss_fns = {signal_type: log_mse_loss for signal_type in
                     self.unique_signal_types}

    # Permutation-invariant training requires realigning the sources to match
    # the references, both in loss computations and computing summary metrics.

    # Build loss split between all-zero and nonzero reference signals.
    self.source_is_nonzero = _weights_for_nonzero_refs(source_waveforms)
    self.source_is_zero = tf.logical_not(self.source_is_nonzero)

    # Waveforms with nonzero references.
    self.source_waveforms_nonzero = tf.boolean_mask(
        source_waveforms, self.source_is_nonzero)[:, tf.newaxis]

    # Waveforms with all-zero references.
    self.source_waveforms_zero = tf.boolean_mask(
        source_waveforms, self.source_is_zero)[:, tf.newaxis]

    self.weight = 1. / tf.cast(batch_size * num_sources, tf.float32)
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
    # Loss for zero references only if self.loss_zero_ref_weight provided.
    if self.loss_zero_ref_weight:
      loss = tf.reduce_sum(
          log_mse_loss(self.source_waveforms_zero,
                       separated_waveforms_zero,
                       max_snr=self.max_snr_for_zero_sources,
                       bias_ref_signal=self.mixture_waveforms_zero))
      loss_zero = tf.identity(self.loss_zero_ref_weight * self.weight * loss,
                              name='loss_ref_zero')
      tf.losses.add_loss(loss_zero)

    # Loss for nonzero references.
    loss = tf.reduce_sum(log_mse_loss(self.source_waveforms_nonzero,
                                      separated_waveforms_nonzero,
                                      max_snr=self.max_snr))
    loss_nonzero = tf.identity(self.weight * loss, name='loss_ref_nonzero')
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
    weights_for_nsrc = tf.logical_and(
        source_is_nonzero,  # Shape (batch, source) indicator of active srcs.
        _weights_for_num_sources(source_waveforms, 1))
    # Compute scalars only for examples with 1 active signal.
    scalars_nsrc, weights_nsrc = summaries.scalar_snr_metrics_weighted(
        new_signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        weights_for_nsrc)
    new_scalars.update({name + '_1src': value
                        for name, value in scalars_nsrc.items()})
    new_weights.update({name + '_1src': value
                        for name, value in weights_nsrc.items()})
    # For min2srcs mixtures, average metric for nonzero reference sources.
    # Compute shape (batch, source) boolean weights for examples with
    # 2+ active signals.
    weights_for_nsrc = tf.logical_and(
        source_is_nonzero,  # Shape (batch, source) indicator of active srcs.
        tf.logical_not(_weights_for_num_sources(source_waveforms, 1)))
    # Compute scalars only for examples with 2+ active signals.
    scalars_nsrc, weights_nsrc = summaries.scalar_snr_metrics_weighted(
        new_signal_names,
        separated_waveforms,
        source_waveforms,
        mixture_waveforms,
        weights_for_nsrc)
    new_scalars.update({name + '_min2srcs': value
                        for name, value in scalars_nsrc.items()})
    new_weights.update({name + '_min2srcs': value
                        for name, value in weights_nsrc.items()})
    # Unless prefix is '' (final separated), exclude duplicate mixture
    # summaries. Also, fix the _signals_ summaries for non-empty prefix.
    if prefix:
      new_audio = {
          k: v for k, v in new_audio.items() if 'mixture_' not in k}
      new_images = {
          k: v for k, v in new_images.items() if 'mixture_' not in k}
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
  loss_names = [loss.name for loss in tf_losses]
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
  if mode != tf.estimator.ModeKeys.PREDICT:
    # Get reference sources at reference mic.
    source_waveforms = features['source_images'][:, :, refmic]
    loss_maker = LossMaker(source_waveforms, hparams.signal_types,
                           hparams.loss_zero_ref_weight,
                           mixture_waveforms_refmic)

  separated_waveforms = None
  # To avoid too many summaries, we only include the separated waveform which
  # has scope_prefix = '' and the last beamformed waveform only.
  prefixes_in_summaries = ['']
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

      if mode != tf.estimator.ModeKeys.PREDICT and not last_iter:

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

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Add loss function on final separated_waveforms.
  separated_waveforms = loss_maker.add_loss(separated_waveforms)

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

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=metrics,
      train_op=train_op,
      training_hooks=[logging_hook])
