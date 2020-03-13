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
"""Tensorflow input/output utilities."""

import os

import tensorflow.compat.v1 as tf




class Features(object):
  """Feature keys."""

  # Waveform(s) of audio observed at receiver(s).
  RECEIVER_AUDIO = 'receiver_audio'

  # Images of each source at each microphone, including reverberation.
  # Images are real valued with shape [sources * microphones, length].
  SOURCE_IMAGES = 'source_images'


def get_inference_spec(num_receivers=1,
                       num_samples=None):
  """Returns a specification of features in tf.Examples in roomsim format."""
  spec = {}
  spec[Features.RECEIVER_AUDIO] = tf.FixedLenFeature(
      [num_receivers, num_samples], tf.float32)

  return spec


def get_roomsim_spec(num_sources,
                     num_receivers,
                     num_samples):
  """Returns a specification of features in tf.Examples in roomsim format.

  Args:
    num_sources: Expected number of sources.
    num_receivers: Number of microphones in array.
    num_samples: Expected length of sources in samples. 'None' for variable.

  Returns:
    Feature specifications suitable to pass to tf.parse_example.
  """
  spec = {}
  spec[Features.RECEIVER_AUDIO] = tf.FixedLenFeature(
      [num_receivers, num_samples], tf.float32)
  spec[Features.SOURCE_IMAGES] = tf.FixedLenFeature(
      [num_sources, num_receivers, num_samples], tf.float32)
  return spec


def placeholders_from_spec(feature_spec):
  """Returns placeholders compatible with a given feature spec."""
  placeholders = {}
  for key, feature in feature_spec.items():
    placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                       shape=[1] + feature.shape,
                                       name=key)
  return placeholders


def read_lines_from_file(file_list_path, skip_fields=0, base_path=None):
  """Read lines from a file.

  Args:
    file_list_path: String specifying absolute path of a file list.
    skip_fields: Skip first n fields in each line of the file list.
    base_path: If not None, use this to build absolute path of files, instead of
        base path of the file list.

  Returns:
    List of strings, which are tab-delimited absolute file paths.
  """
  # Read in and preprocess the file list.
  with open(file_list_path, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line.split('\t')[skip_fields:] for line in lines]

  # Make each relative path point to an absolute path.
  lines_abs_path = []
  if base_path is None:
    base_path = os.path.dirname(file_list_path)
  for line in lines:
    wavs_abs_path = []
    for wav in line:
      wavs_abs_path.append(os.path.join(base_path, wav))
    lines_abs_path.append(wavs_abs_path)
  lines = lines_abs_path

  # Rejoin the fields to return a list of strings.
  return ['\t'.join(fields) for fields in lines]


def wavs_to_dataset(file_list,
                    batch_size,
                    num_samples,
                    parallel_readers=1,
                    randomize_order=False,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True):
  """Fetches features from list of wav files.

  Args:
    file_list: List of tab-delimited file locations of wavs. Each line should
        correspond to one example, where each field is a source wav.
    batch_size: The number of examples to read.
    num_samples: Number of samples in each wav file.
    parallel_readers: Number of fetches that should happen in parallel.
    randomize_order: Whether to randomly shuffle features.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.

  Returns:
    A batch_size number of features constructed from wav files.
  """
  lines = [line.split('\t') for line in file_list]

  # Examples that have fewer than max_sources sources will be padded with zeros.
  max_sources = max([len(line) for line in lines])
  lines = [line + ['0'] * (max_sources - len(line)) for line in lines]
  wavs = []
  for line in lines:
    for wav in line:
      wavs.append(wav)
  wavs = tf.constant(wavs)

  def decode_wav(wav):
    audio_bytes = tf.read_file(wav)
    waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1,
                                      desired_samples=num_samples)
    waveform = tf.reshape(waveform, (1, num_samples))
    return waveform

  def decode_wav_or_return_zeros(wav):
    return tf.cond(tf.equal(wav, '0'),
                   lambda: tf.zeros((1, num_samples), dtype=tf.float32),
                   lambda: decode_wav(wav))

  def combine_mixture_and_sources(waveforms):
    # waveforms is shape (max_sources, 1, num_samples).
    mixture_waveform = tf.reduce_sum(waveforms, axis=0)
    source_waveforms = tf.reshape(waveforms, (max_sources, 1, num_samples))
    return {'receiver_audio': mixture_waveform,
            'source_images': source_waveforms}

  dataset = tf.data.Dataset.from_tensor_slices(wavs)
  dataset = dataset.map(decode_wav_or_return_zeros)
  dataset = dataset.batch(max_sources)
  dataset = dataset.map(combine_mixture_and_sources)

  if randomize_order:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.prefetch(parallel_readers)
  dataset = dataset.take(num_examples)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  if repeat:
    dataset = dataset.repeat()

  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def input_fn(params):
  """An input function that uses params['feature_spec'].

  Args:
    params: A dictionary of experiment params.

  Returns:
    Features specified by params['feature_spec'].  If 'inference' exists and is
    True in params, then placeholders will be returned based on the spec in
    params['inference_spec'], otherwise a dataset of examples read from
    params['input_data'] will be returned.
  """
  if params.get('inference', False):
    feature_spec = params['inference_spec']
    with tf.variable_scope('input_audio'):
      return placeholders_from_spec(feature_spec)
  else:
    file_list = params.get('input_data', None)
    if not isinstance(file_list, list):
      file_list = read_lines_from_file(file_list, skip_fields=1)
    batch_size = params.get('batch_size', None)
    io_params = params.get('io_params', {})
    return wavs_to_dataset(file_list,
                           batch_size,
                           **io_params)
