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

import argparse
import os
import sys
cur_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, 'dcase2020_fuss_baseline'))

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import inference
from train import data_io
from train import metrics
from train import permutation_invariant


def compute_sisnri(source_waveforms, separated_waveforms, mixture_waveform):
  """Compute scale-invariant SNR."""
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  num_sources = source_waveforms.shape[0]
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(mixture_waveform[tf.newaxis], (num_sources, 1)),
      source_waveforms)
  return sisnr_separated, sisnr_mixture


def _print_score_stats(sisnri_per_source, label=''):
  values_all = []
  for idx, values in sisnri_per_source.items():
    print('SI-SNR%s for source %s = %.1f +/- %.1f dB' % (
        label, idx, np.mean(values), np.std(values)))
    values_all.extend(list(values))
  print('Overall SI-SNR%s = %.1f +/- %.1f dB' % (
      label, np.mean(values_all), np.std(values_all)))


def main():
  parser = argparse.ArgumentParser(
      description='Evaluate a source separation model.')
  parser.add_argument(
      '-cp', '--checkpoint_path', help='Path for model checkpoint files.',
      required=True)
  parser.add_argument(
      '-mp', '--metagraph_path', help='Path for inference metagraph.',
      required=True)
  parser.add_argument(
      '-dp', '--data_list_path', help='Path for list of files.',
      required=True)
  parser.add_argument(
      '-op', '--output_path', help='Path of resulting csv file.',
      required=True)
  args = parser.parse_args()

  model = inference.SeparationModel(args.checkpoint_path,
                                    args.metagraph_path)

  file_list = data_io.read_lines_from_file(args.data_list_path, skip_fields=1,
                                           base_path=None)
  with model.graph.as_default():
    dataset = data_io.wavs_to_dataset(file_list, batch_size=1,
                                      num_samples=160000,
                                      repeat=False,
                                      combine_by_class=True,
                                      fixed_classes=['BG_DSD', 'FG_DSD',
                                                     'FUSS_mixture'])
    # Strip batch and mic dimensions.
    dataset['receiver_audio'] = dataset['receiver_audio'][0, 0]
    dataset['source_images'] = dataset['source_images'][0, :, 0]

  # Separate with a trained model.
  i = 1
  num_sources = 3
  sisnr_per_source = {c: [] for c in range(num_sources)}
  sisnri_per_source = {c: [] for c in range(num_sources)}
  columns_mix = ['SISNR_mixture_source%d' % j
                 for j in range(num_sources)]
  columns_sep = ['SISNR_separated_source%d' % j
                 for j in range(num_sources)]
  df = pd.DataFrame(columns=columns_mix + columns_sep)
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
    sisnr_sep, sisnr_mix = compute_sisnri(source_waveforms,
                                          separated_waveforms,
                                          waveforms['receiver_audio'])
    sisnr_sep = sisnr_sep.numpy()
    sisnr_mix = sisnr_mix.numpy()
    sisnr_imp = sisnr_sep - sisnr_mix

    row_dict = {col: sisnr for col, sisnr
                in zip(columns_mix[:len(sisnr_mix)], sisnr_mix)}
    row_dict.update({col: sisnr for col, sisnr
                     in zip(columns_sep[:len(sisnr_sep)], sisnr_sep)})
    new_row = pd.Series(row_dict)
    df = df.append(new_row, ignore_index=True)
    for j in range(num_sources):
      sisnr_per_source[j].append(sisnr_sep[j])
      sisnri_per_source[j].append(sisnr_imp[j])
    print('Example %d: SI-SNR sep = %.1f dB, SI-SNR mix = %.1f dB,'
          'SI-SNR imp = %.1f dB' % (
              i, np.mean(sisnr_sep), np.mean(sisnr_mix),
              np.mean(sisnr_sep - sisnr_mix)))
    if not i % 20:
      # Report mean statistics every so often.
      _print_score_stats(sisnr_per_source)
      _print_score_stats(sisnri_per_source, ' imp')
    i += 1

  # Report final mean statistics.
  print('\nFinal statistics:')
  _print_score_stats(sisnr_per_source)
  _print_score_stats(sisnri_per_source, ' imp')

  # Write csv.
  csv_path = os.path.join(args.output_path, 'scores.csv')
  print('\nWriting csv to %s.' % csv_path)
  df.to_csv(csv_path)


if __name__ == '__main__':
  main()
