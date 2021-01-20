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
"""Evaluate separated audio from a DCASE 2020 task 4 separation model."""

import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import inference
from train import data_io
from train import metrics
from train import permutation_invariant


def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (source,) weights for signals that are nonzero."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  return tf.greater(source_norms, 1e-8)


def _weights_for_active_seps(power_sources, power_separated):
  """Return (source,) weights for active separated signals."""
  min_power = tf.reduce_min(power_sources, axis=-1, keepdims=True)
  return tf.greater(power_separated, 0.01 * min_power)


def compute_metrics(source_waveforms, separated_waveforms, mixture_waveform):
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

  # Align separated sources to reference sources.
  perm_inv_loss = permutation_invariant.wrap(
      lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar))
  _, separated_waveforms = perm_inv_loss(source_waveforms[tf.newaxis],
                                         separated_waveforms[tf.newaxis])
  separated_waveforms = separated_waveforms[0]  # Remove batch axis.

  # Compute separated and source powers.
  power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
  power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

  # Compute weights for active (separated, source) pairs where source is nonzero
  # and separated power is above threshold of quietest source power - 20 dB.
  weights_active_refs = _weights_for_nonzero_refs(source_waveforms)
  weights_active_seps = _weights_for_active_seps(
      tf.boolean_mask(power_sources, weights_active_refs), power_separated)
  weights_active_pairs = tf.logical_and(weights_active_refs,
                                        weights_active_seps)

  # Compute SI-SNR.
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
  num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))
  num_active_pairs = tf.reduce_sum(tf.cast(weights_active_pairs, tf.int32))
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(mixture_waveform[tf.newaxis], (source_waveforms.shape[0], 1)),
      source_waveforms)

  # Compute under/equal/over separation.
  under_separation = tf.cast(tf.less(num_active_seps, num_active_refs),
                             tf.float32)
  equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs),
                             tf.float32)
  over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs),
                            tf.float32)

  return {'sisnr_separated': sisnr_separated,
          'sisnr_mixture': sisnr_mixture,
          'sisnr_improvement': sisnr_separated - sisnr_mixture,
          'power_separated': power_separated,
          'power_sources': power_sources,
          'under_separation': under_separation,
          'equal_separation': equal_separation,
          'over_separation': over_separation,
          'weights_active_refs': weights_active_refs,
          'weights_active_seps': weights_active_seps,
          'weights_active_pairs': weights_active_pairs,
          'num_active_refs': num_active_refs,
          'num_active_seps': num_active_seps,
          'num_active_pairs': num_active_pairs}


def _report_score_stats(metric_per_source_count, label='', counts=None):
  """Report mean and std dev for specified counts."""
  values_all = []
  if counts is None:
    counts = metric_per_source_count.keys()
  for count in counts:
    values = metric_per_source_count[count]
    values_all.extend(list(values))
  return '%s for count(s) %s = %.1f +/- %.1f dB' % (
      label, counts, np.mean(values_all), np.std(values_all))


def evaluate(checkpoint_path, metagraph_path, data_list_path, output_path):
  """Evaluate a model on FUSS data."""
  model = inference.SeparationModel(checkpoint_path, metagraph_path)

  file_list = data_io.read_lines_from_file(data_list_path, skip_fields=1)
  with model.graph.as_default():
    dataset = data_io.wavs_to_dataset(file_list, batch_size=1,
                                      num_samples=160000,
                                      repeat=False)
    # Strip batch and mic dimensions.
    dataset['receiver_audio'] = dataset['receiver_audio'][0, 0]
    dataset['source_images'] = dataset['source_images'][0, :, 0]

  # Separate with a trained model.
  i = 1
  max_count = 4
  dict_per_source_count = lambda: {c: [] for c in range(1, max_count + 1)}
  sisnr_per_source_count = dict_per_source_count()
  sisnri_per_source_count = dict_per_source_count()
  under_seps = []
  equal_seps = []
  over_seps = []
  df = None
  while True:
    try:
      waveforms = model.sess.run(dataset)
    except tf.errors.OutOfRangeError:
      break
    separated_waveforms = model.separate(waveforms['receiver_audio'])
    source_waveforms = waveforms['source_images']
    if np.allclose(source_waveforms, 0):
      print('WARNING: all-zeros source_waveforms tensor encountered.'
            'Skiping this example...')
      continue
    metrics_dict = compute_metrics(source_waveforms, separated_waveforms,
                                   waveforms['receiver_audio'])
    metrics_dict = {k: v.numpy() for k, v in metrics_dict.items()}
    sisnr_sep = metrics_dict['sisnr_separated']
    sisnr_mix = metrics_dict['sisnr_mixture']
    sisnr_imp = metrics_dict['sisnr_improvement']
    weights_active_pairs = metrics_dict['weights_active_pairs']

    # Create and initialize the dataframe if it doesn't exist.
    if df is None:
      # Need to create the dataframe.
      columns = []
      for metric_name, metric_value in metrics_dict.items():
        if metric_value.shape:
          # Per-source metric.
          for i_src in range(1, max_count + 1):
            columns.append(metric_name + '_source%d' % i_src)
        else:
          # Scalar metric.
          columns.append(metric_name)
      columns.sort()
      df = pd.DataFrame(columns=columns)
      if output_path.endswith('.csv'):
        csv_path = output_path
      else:
        csv_path = os.path.join(output_path, 'scores.csv')

    # Update dataframe with new metrics.
    row_dict = {}
    for metric_name, metric_value in metrics_dict.items():
      if metric_value.shape:
        # Per-source metric.
        for i_src in range(1, max_count + 1):
          row_dict[metric_name + '_source%d' % i_src] = metric_value[i_src - 1]
      else:
        # Scalar metric.
        row_dict[metric_name] = metric_value
    new_row = pd.Series(row_dict)
    df = df.append(new_row, ignore_index=True)

    # Store metrics per source count and report results so far.
    under_seps.append(metrics_dict['under_separation'])
    equal_seps.append(metrics_dict['equal_separation'])
    over_seps.append(metrics_dict['over_separation'])
    sisnr_per_source_count[metrics_dict['num_active_refs']].extend(
        sisnr_sep[weights_active_pairs].tolist())
    sisnri_per_source_count[metrics_dict['num_active_refs']].extend(
        sisnr_imp[weights_active_pairs].tolist())
    print('Example %d: SI-SNR sep = %.1f dB, SI-SNR mix = %.1f dB, '
          'SI-SNR imp = %.1f dB, ref count = %d, sep count = %d' % (
              i, np.mean(sisnr_sep), np.mean(sisnr_mix),
              np.mean(sisnr_sep - sisnr_mix), metrics_dict['num_active_refs'],
              metrics_dict['num_active_seps']))
    if not i % 20:
      # Report mean statistics and save csv every so often.
      lines = [
          'Metrics after %d examples:' % i,
          _report_score_stats(sisnr_per_source_count, 'SI-SNR',
                              counts=[1]),
          _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                              counts=[2]),
          _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                              counts=[3]),
          _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                              counts=[4]),
          _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                              counts=[2, 3, 4]),
          'Under separation: %.2f' % np.mean(under_seps),
          'Equal separation: %.2f' % np.mean(equal_seps),
          'Over separation: %.2f' % np.mean(over_seps),
          ]
      print('')
      for line in lines:
        print(line)
      with open(csv_path.replace('.csv', '_summary.txt'), 'w+') as f:
        f.writelines([line + '\n' for line in lines])

      print('\nWriting csv to %s.\n' % csv_path)
      df.to_csv(csv_path)
    i += 1

  # Report final mean statistics.
  lines = [
      'Final statistics:',
      _report_score_stats(sisnr_per_source_count, 'SI-SNR',
                          counts=[1]),
      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                          counts=[2]),
      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                          counts=[3]),
      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                          counts=[4]),
      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                          counts=[2, 3, 4]),
      'Under separation: %.2f' % np.mean(under_seps),
      'Equal separation: %.2f' % np.mean(equal_seps),
      'Over separation: %.2f' % np.mean(over_seps),
      ]
  print('')
  for line in lines:
    print(line)
  with open(csv_path.replace('.csv', '_summary.txt'), 'w+') as f:
    f.writelines([line + '\n' for line in lines])

  # Write final csv.
  print('\nWriting csv to %s.' % csv_path)
  df.to_csv(csv_path)
