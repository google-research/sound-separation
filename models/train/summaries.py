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
"""Compute summaries for a denoising model.

Functions in this file only compute summary tensors, but do not create any
actual summaries.  Training summaries can be created using summary_util.

summary_dict = summaries.compute_summaries(...)
summary_util.create_summaries(step, **summary_dict)

"""

import tensorflow.compat.v1 as tf

from . import metrics
from . import signal_util


def get_statistics(values):
  """Computes mean and standard deviation.

  Args:
    values: a Tensor with values to summarize.

  Returns:
     A dictionary of names and scalar tensors.
  """
  size = tf.reduce_sum(tf.ones_like(values))
  size = tf.maximum(size, tf.ones_like(size))  # Avoid divide-by-zero
  mean = tf.reduce_sum(values) / size

  return {'mean': mean}


def get_statistics_from_dict(batched_metrics):
  """Returns a dictionary of statistics given dictionary of batch of values."""
  scalars = {}
  for name, values in batched_metrics.items():
    for stat_name, value in get_statistics(values).items():
      scalars['%s_%s' % (name, stat_name)] = value
  return scalars


def spectrogram_summary(name, spectrogram_batch):
  """Display a spectrogram (or spectrogram mask) as a TensorFlow image summary.

  With multi-channel data, we show a summary for each channel in
  `name`/channel0, etc. named summaries.

  Args:
    name: Name of the TensorFlow image summary.
    spectrogram_batch: Batch of spectrograms (or masks, etc.), of shape
        (batch_size, frames, bins) or (batch_size, frames, bins, channels)

  Returns:
    Dictionary of image summaries.
  """
  if spectrogram_batch.shape.ndims == 3:
    # Go from (batch,             frames, bins)
    # to      (batch, channels=1, frames, bins)
    spectrogram_batch = tf.expand_dims(spectrogram_batch, 1)

  assert spectrogram_batch.shape.ndims == 4, spectrogram_batch.shape
  num_channels = spectrogram_batch.shape[1]
  assert num_channels < 10, 'TensorBoard insanity!'

  image_summaries = {}
  for i in range(num_channels):
    # This takes us from (batch, channels, frames, bins) to
    # (batch, bins, frames, 1).
    single_channel = tf.transpose(
        spectrogram_batch[:, i:i+1], [0, 3, 2, 1])
    channel_name = '%s/channel%d' % (name, i) if num_channels > 1 else name

    # We reverse bins to display low freqs at bottom instead of top.
    image_summaries[channel_name] = tf.reverse_v2(single_channel, [1])

  return image_summaries


def compute_spectrogram_summaries(signal_names,
                                  separated_spectrograms,
                                  source_spectrograms,
                                  mixture_spectrograms):
  """Computes image summaries for given spectrograms.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_spectrograms: Tensor of STFTs of the separated output signals.
    source_spectrograms: Tensor of STFTs of source (training target) signals.
    mixture_spectrograms: Tensor of STFTs of the mixture input signals.

  Returns:
    Dictionary of image summaries.
  """
  spectrogram_summaries = {}
  if mixture_spectrograms is not None:
    spectrogram_summaries[
        'mixture'] = signal_util.stabilized_power_compress_abs(
            mixture_spectrograms, 0.3, 1e-8)

  for k, signal_name in enumerate(signal_names):
    # Write images of spectrograms.
    if source_spectrograms is not None:
      spectrogram_summaries['source_%s_spectrogram' % signal_name] = (
          signal_util.stabilized_power_compress_abs(
              source_spectrograms[:, k], 0.3, 1e-8))
    if separated_spectrograms is not None:
      spectrogram_summaries['separated_%s_spectrogram' % signal_name] = (
          signal_util.stabilized_power_compress_abs(
              separated_spectrograms[:, k], 0.3, 1e-8))

  image_summaries = {}
  for name, value in spectrogram_summaries.items():
    image_summaries.update(spectrogram_summary(name, value))

  return image_summaries


def compute_audio_summaries(signal_names,
                            separated_waveforms,
                            source_waveforms,
                            mixture_waveforms):
  """Computes audio summaries for given waveforms.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_waveforms: The separated signals, of shape
        [batch, numsources, samples].
    source_waveforms: The source signals (training targets), of shape
        [batch, numsources, samples].
    mixture_waveforms: The mixture input signals, of shape
        [batch, numreceivers, samples].

  Returns:
    Dictionary of audio summaries.
  """
  audio_summaries = {}
  if mixture_waveforms is not None:
    audio_summaries['mixture'] = tf.transpose(mixture_waveforms[:, :2],
                                              [0, 2, 1])

  for k, name in enumerate(signal_names):
    if separated_waveforms is not None:
      audio_summaries['separated_%s_audio' % name] = separated_waveforms[:, k]
    if source_waveforms is not None:
      audio_summaries['source_%s_audio' % name] = source_waveforms[:, k]
  return audio_summaries


def compute_loss_summaries(loss_names, loss_tensor):
  """Computes loss summaries as a dictionary of scalar tensors.

  Args:
    loss_names: A list of strings of loss names of nlosses length.
    loss_tensor: A tensor of shape [1, nlosses] which contains the losses.
  Returns:
    loss_summary_dict: A dictionary of loss summaries.
  """
  loss_summary_dict = {}
  loss_count = 0
  loss_list = tf.unstack(loss_tensor[0])
  assert len(loss_list) == len(loss_names)
  total_loss = tf.constant(0.0, tf.float32)
  for loss_name, loss in zip(loss_names, loss_list):
    loss_count += 1
    # There is an inconsistency of loss summary names between train and eval.
    # Eval summaries come out as loss_name:0, train summaries as loss_name_0.
    # So, we change : to _ in the loss.name to avoid having different names.
    loss_name = loss_name.replace(':', '_', 1)
    loss_summary_dict[loss_name] = loss
    total_loss += loss
  if loss_count >= 2:
    loss_summary_dict['loss/total_loss'] = total_loss
  return loss_summary_dict


def _get_snr_metrics_dict(signal_names,
                          separated_waveforms,
                          source_waveforms,
                          mixture_waveforms,
                          metric_functions):
  """Computes dict of SNR metrics.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_waveforms: The separated signals, of shape
        (batch, source, time).
    source_waveforms: The source signals (training targets), of shape
        (batch, source, time).
    mixture_waveforms: The mixture input signals, of shape
        (batch, mic, time).
    metric_functions: A dict of string->function pairs mapping metric name to
        metric function with signature metric_fn(estimated, source).

  Returns:
    A dict of string->tf.Tensor pairs mapping metric name to metric values.
  """
  metrics_dict = {}
  for k, signal_name in enumerate(signal_names):
    for metric_name, metric_fn in metric_functions.items():
      mixture = metric_fn(mixture_waveforms[:, 0], source_waveforms[:, k])
      separated = metric_fn(separated_waveforms[:, k], source_waveforms[:, k])
      improv = separated - mixture
      metrics_dict['SNR/%s/mixture_%s' % (metric_name, signal_name)] = mixture
      metrics_dict[
          'SNR/%s/separated_%s' % (metric_name, signal_name)] = separated
      metrics_dict[
          'SNR/%s/improvement_%s' % (metric_name, signal_name)] = improv

  return metrics_dict


def _aggregate_metrics_over_sources(signal_names, metrics_dict):
  """Updates metrics to include aggregations of values over all signals.

  Args:
    signal_names: A list of signal names.
    metrics_dict: A dictionary with keys that are metric_names in the form of
      'metric/quantity_signal1' and with a value a tensor of [batch]
      dimension.
  Returns:
    Nothing, but updates batched_metrics to include aggregations over
      signal1, signal2, ... and aggregate all their entries into a signals
      entry in the form of 'metric/quantity_signals'.
  """
  extras = {}
  for name, value in metrics_dict.items():
    for signal_name in signal_names:
      if name.endswith(signal_name):
        base_name = name[0:-len(signal_name)] + 'signals'
        if base_name not in extras:
          extras[base_name] = []
        extras[base_name].append(value)
  metrics_dict.update(
      {key: tf.concat(value, 0) for key, value in extras.items()})


def _apply_weights_to_metrics(metrics_dict, weights, signal_names):
  """Apply weights to metrics dictionary.

  Args:
    metrics_dict: A dictionary with keys that are metric_names in the form of
      'metric/quantity_signal1' with value tf.Tensor of shape (batch,), or
      'metric/quantity_signals' with value tf.Tensor of shape (batch * source,).
    weights: a tf.Tensor of shape (batch, source).
    signal_names: List of signal names.

  Returns:
    A dict of scalar statistics.
    A dict of weights per scalar statistic, suitable as weights for tf.metrics.

  Raises:
    ValueError: If metrics_dict contains a key that does not contain 'signals'
      or a string from signal_names.
  """
  metrics_weighted_dict = {}
  weights_dict = {}
  # Update metrics_dict of aggregated values with weights.
  for metric_name, metric_value in metrics_dict.items():
    if metric_name.endswith('signals'):
      # Flatten into shape (batch * source,).
      mask = tf.reshape(tf.transpose(weights, (1, 0)), (-1,))
    else:
      for k, signal_name in enumerate(signal_names):
        if signal_name in metric_name:
          idx_signal = k
          break
      else:
        raise ValueError('Encountered metric_name {!r} that does not contain'
                         'signals or any of {}'.format(metric_name,
                                                       signal_names))
      mask = weights[:, idx_signal]
    metrics_weighted_dict[metric_name] = tf.boolean_mask(metric_value, mask)
    weights_dict[metric_name] = tf.reduce_mean(tf.cast(mask, tf.float32))

  return metrics_weighted_dict, weights_dict


def scalar_snr_metrics_weighted(signal_names,
                                separated_waveforms,
                                source_waveforms,
                                mixture_waveforms,
                                weights=None):
  """Compute weighted tensorflow SNR summaries for a separation model.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_waveforms: tf.Tensor of separated signals, of shape
        (batch, source, time).
    source_waveforms: tf.Tensor of source signals (training targets), of shape
        (batch, source, time).
    mixture_waveforms: tf.Tensor of mixture input signals, of shape
        (batch, mic, time).
    weights: tf.Tensor of dtype bool with shape (batch,) or (batch, source)
        used to compute metric statistics on a subset of data.

  Returns:
    A dict of scalar statistics.
    A dict of weights per scalar statistic, suitable as tf.metrics weights. Is
       an empty dict if weights is None.
  """
  metric_functions = {
      'snr_residual': metrics.signal_to_noise_ratio_residual,
      'snr_gain_invariant': metrics.signal_to_noise_ratio_gain_invariant,
  }

  metrics_dict = _get_snr_metrics_dict(signal_names,
                                       separated_waveforms,
                                       source_waveforms,
                                       mixture_waveforms,
                                       metric_functions)

  _aggregate_metrics_over_sources(signal_names, metrics_dict)

  weights_dict_for_stats = {}
  if weights is not None:
    metrics_dict, weights_dict = _apply_weights_to_metrics(metrics_dict,
                                                           weights,
                                                           signal_names)

    for metric_name in weights_dict:
      for stat in ['_mean']:
        weights_dict_for_stats[metric_name + stat] = weights_dict[metric_name]
  return get_statistics_from_dict(metrics_dict), weights_dict_for_stats


def scalar_snr_metrics(signal_names,
                       separated_waveforms,
                       source_waveforms,
                       mixture_waveforms):
  """Compute tensorflow SNR summaries for a separation model.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_waveforms: tf.Tensor of separated signals, of shape
        (batch, source, time).
    source_waveforms: tf.Tensor of source signals (training targets), of shape
        (batch, source, time).
    mixture_waveforms: tf.Tensor of mixture input signals, of shape
        (batch, mic, time).

  Returns:
    A dictionary of scalar statistics.
  """
  scalars, _ = scalar_snr_metrics_weighted(signal_names, separated_waveforms,
                                           source_waveforms, mixture_waveforms,
                                           weights=None)
  return scalars


def additional_scalar_metrics(additional_tensors_dict):
  """Compute additional TF scalar summaries.

  Args:
    additional_tensors_dict: A dict of additional scalar tensors to summarize.

  Returns:
    A dictionary of scalar statistics.
  """
  tensors = {}
  for key, value in additional_tensors_dict.items():
    tensors['additional_tensors/%s' % key] = value

  return get_statistics_from_dict(tensors)


def compute_summaries(signal_names,
                      separated_spectrograms,
                      source_spectrograms,
                      mixture_spectrograms,
                      separated_waveforms,
                      source_waveforms,
                      mixture_waveforms,
                      learning_rate=None,
                      additional_summary_tensors=None):
  """Compute TF summaries for a denoising model.

  Args:
    signal_names: List of signal names for which to compute summaries.
    separated_spectrograms: Tensor of STFTs of the separated output signals.
    source_spectrograms: Tensor of STFTs of source (training target) signals.
    mixture_spectrograms: Tensor of STFTs of the mixture input signals.
    separated_waveforms: The separated signals, of shape
        [batch, numsources, samples].
    source_waveforms: The source signals (training targets), of shape
        [batch, numsources, samples].
    mixture_waveforms: The mixture input signals, of shape
        [batch, numreceivers, samples].
    learning_rate: Learning rate, or None.
    additional_summary_tensors: A dictionary of additional tensors to compute
         statistical summaries for.

  Returns:
    Dictionary of summaries, containing:
      - scalars: Dictionary with names and scalar tensors for summarization.
      - audio: Dictionary with names and audio tensors for summarization.
      - images: Dictionary with names and image tensors for summarization.
  """
  assert len(signal_names) == separated_spectrograms.shape[1]
  assert len(signal_names) == source_spectrograms.shape[1]

  summaries = {
      'scalars': {},
      'audio': {},
      'images': {},
  }
  summaries['images'].update(
      compute_spectrogram_summaries(
          signal_names,
          separated_spectrograms,
          source_spectrograms,
          mixture_spectrograms))

  summaries['audio'].update(
      compute_audio_summaries(
          signal_names,
          separated_waveforms,
          source_waveforms,
          mixture_waveforms))

  summaries['scalars'].update(
      scalar_snr_metrics(
          signal_names,
          separated_waveforms,
          source_waveforms,
          mixture_waveforms))

  if additional_summary_tensors:
    tensors = {}
    for key, value in additional_summary_tensors.items():
      tensors['additional_tensors/%s' % key] = value
    summaries['scalars'].update(get_statistics_from_dict(tensors))

  tf_losses = tf.losses.get_losses()
  loss_names = [loss.name for loss in tf_losses]
  loss_tensor = tf.expand_dims(tf.stack(tf_losses), axis=0)
  summaries['scalars'].update(compute_loss_summaries(loss_names, loss_tensor))

  if learning_rate is not None:
    learning_rate_summary = {'learning_rate': learning_rate}
    summaries['scalars'].update(learning_rate_summary)

  return summaries
