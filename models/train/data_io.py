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

import collections
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


def read_lines_from_file(file_list_path, skip_fields=0, base_path='relative'):
  """Read lines from a file.

  Args:
    file_list_path: String specifying absolute path of a file list.
    skip_fields: Skip first n fields in each line of the file list.
    base_path: If `relative`, use base path of file list. If None, don't add
        any base path. If not None, use base_path to build absolute paths.

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
  if base_path == 'relative':
    base_path = os.path.dirname(file_list_path)
  elif base_path is None:
    base_path = ''
  for line in lines:
    wavs_abs_path = []
    for wav in line:
      wavs_abs_path.append(os.path.join(base_path, wav))
    lines_abs_path.append(wavs_abs_path)
  lines = lines_abs_path

  # Rejoin the fields to return a list of strings.
  return ['\t'.join(fields) for fields in lines]


def unique_classes_from_lines(lines):
  """Return sorted list of unique classes that occur in all lines."""
  # Build sorted list of unique classes.
  unique_classes = sorted(
      list({x.split(':')[0] for line in lines for x in line}))  # pylint: disable=g-complex-comprehension
  return unique_classes


def wavs_to_dataset(file_list,
                    batch_size,
                    num_samples,
                    parallel_readers=1,
                    randomize_order=False,
                    combine_by_class=False,
                    fixed_classes=None,
                    max_sources_override=None,
                    num_examples=-1,
                    shuffle_buffer_size=50,
                    repeat=True,
                    num_mics=1):
  r"""Fetches features from list of wav files.

  Args:
    file_list: List of tab-delimited file locations of wavs. Each line should
        correspond to one example, where each field is a source wav.
    batch_size: The number of examples to read.
    num_samples: Number of samples in each wav file.
    parallel_readers: Number of fetches that should happen in parallel.
    randomize_order: Whether to randomly shuffle features.
    combine_by_class: Whether to add together events of the same class.
        Note that this assumes the file list has class annotations, in format:
        '<class 1>:<filename 1>\t<class 2>:<filename 2>
        The number of output sources N depends on fixed_classes:

        If fixed_classes contains all unique classes, N will be the number of
        unique classes in the file list. Each class will have a fixed output
        index, where the order of indices is order of fixed_classes.

        If fixed_classes contains a subset of unique classes, N will be number
        of fixed classes plus maximum number of nonfixed classes in any line
        in the file. For example, if a dataset contains unique classes 'dog',
        'cat', 'parrot', and fixed_classes is ['dog'], and every line only
        contains the classes ['dog', 'cat'] or ['dog', 'parrot'], then the
        number of output sources will be 2, and the 'dog' class will always be
        output at source index 0. If there are M fixed_classes, the first M
        sources will be fixed, and the remaining N - M sources will be nonfixed.

        If fixed_classes is empty, N will be the maximum number of
        unique class occurrences in any line in the file.

    fixed_classes: List of classes to place at fixed source indices.
        Classes that are not in these keys are placed in remaining source
        indices in the order they appear in the file list line.
    max_sources_override: Override maximum number of output sources. Raises
        error if this number is less than assumed max number of sources N.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.
    num_mics: The expected number of mics in source wav files.

  Returns:
    A batch_size number of features constructed from wav files.

  Raises:
    ValueError if max_sources_override is less than assumed max number sources.
  """
  if fixed_classes is None:
    fixed_classes = []

  lines = [line.split('\t') for line in file_list]
  max_component_sources = max([len(line) for line in lines])

  if not combine_by_class:
    # Not combining by class.
    max_combined_sources = max_component_sources

  else:
    # Combine sources of the same class. Assumes each field in the tab-delimited
    # line is <class name>:<wav filename>.

    # Get unique classes.
    unique_classes = unique_classes_from_lines(lines)

    # Build map from class to index for fixed sources.
    fixed_class_to_id = {}
    fixed_classes = [c for c in fixed_classes if c in unique_classes]
    fixed_class_to_id.update({c: i for (i, c)  in enumerate(fixed_classes)})

    # Find maximum number of unique nonfixed classes in any line.
    max_unique_nonfixed_classes = max(
        [len({x.split(':')[0] for x in line}.difference(set(fixed_classes)))
         for line in lines])

    max_combined_sources = len(fixed_classes) + max_unique_nonfixed_classes

    # Override max sources, if specified.
    if max_sources_override:
      if max_sources_override > max_combined_sources:
        max_combined_sources = max_sources_override
      elif max_sources_override < max_combined_sources:
        raise ValueError('max_sources_override of {} is less than assumed max'
                         'combined sources of {}.'.format(max_sources_override,
                                                          max_combined_sources))

    wav_filenames = []
    class_id_list = []
    for line in lines:
      # Extract classes and wav filenames.
      line = [class_and_wav.split(':') for class_and_wav in line]
      wav_classes = [class_and_wav[0] for class_and_wav in line]
      wav_filenames.append([class_and_wav[1] for class_and_wav in line])

      # Get unique list of classes ordered by first appearance in this line.
      unique_classes_by_appearance = list(
          collections.OrderedDict.fromkeys(wav_classes))
      if fixed_classes:
        # Find unique nonfixed classes in this line.
        classes_nonfixed = [c for c in unique_classes_by_appearance
                            if c not in fixed_classes]
        class_ids_nonfixed = range(len(fixed_classes),
                                   len(fixed_classes) + len(classes_nonfixed))
        # Build map of classes to class index for this example.
        class_to_id = dict(fixed_class_to_id)
        class_to_id.update({c: i for (c, i) in zip(classes_nonfixed,
                                                   class_ids_nonfixed)})
      else:
        # Get unique list of classes ordered by first appearance in this line.
        classes = list(collections.OrderedDict.fromkeys(wav_classes))
        # Map classes to class index within this example.
        class_to_id = {c: i for (i, c) in enumerate(classes)}

      wav_class_ids = [class_to_id[c] for c in wav_classes]
      class_id_list.append(wav_class_ids)

    # Pad class id_list so the array is rectangular, so it can be a Tensor.
    class_id_list = [class_ids + [-1] * (max_component_sources - len(class_ids))
                     for class_ids in class_id_list]
    lines = wav_filenames

  # Examples that have fewer than max_component_sources are padded with zeros.
  lines = [line + ['0'] * (max_component_sources - len(line)) for line in lines]

  wavs = []
  for line in lines:
    for wav in line:
      wavs.append(wav)
  wavs = tf.constant(wavs)

  # Read in wav files.
  def decode_wav(wav):
    audio_bytes = tf.read_file(wav)
    waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=num_mics,
                                      desired_samples=num_samples)
    waveform = tf.transpose(waveform)
    waveform = tf.reshape(waveform, (num_mics, num_samples))
    return waveform

  def decode_wav_or_return_zeros(wav):
    return tf.cond(tf.equal(wav, '0'),
                   lambda: tf.zeros((num_mics, num_samples), dtype=tf.float32),
                   lambda: decode_wav(wav))

  dataset = tf.data.Dataset.from_tensor_slices(wavs)
  dataset = dataset.map(decode_wav_or_return_zeros)
  dataset = dataset.batch(max_component_sources)

  # Combine by class, if specified.
  if combine_by_class:
    class_id_list = tf.constant(class_id_list)
    class_id_dataset = tf.data.Dataset.from_tensor_slices(class_id_list)
    dataset = tf.data.Dataset.zip((dataset, class_id_dataset))
    def combine_sources_by_class(wavs, class_ids):
      return tf.math.unsorted_segment_sum(wavs, class_ids, max_combined_sources)
    dataset = dataset.map(combine_sources_by_class)

  # Build mixture and sources waveforms.
  def combine_mixture_and_sources(waveforms):
    # waveforms is shape (max_combined_sources, num_mics, num_samples).
    mixture_waveform = tf.reduce_sum(waveforms, axis=0)
    source_waveforms = tf.reshape(waveforms,
                                  (max_combined_sources, num_mics, num_samples))
    return {'receiver_audio': mixture_waveform,
            'source_images': source_waveforms}
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
    io_params = params.get('io_params', {})
    if io_params.get('combine_by_class', False):
      base_path = None
    else:
      base_path = 'relative'
    if not isinstance(file_list, list):
      file_list = read_lines_from_file(file_list, skip_fields=1,
                                       base_path=base_path)
    batch_size = params.get('batch_size', None)
    return wavs_to_dataset(file_list,
                           batch_size,
                           **io_params)
