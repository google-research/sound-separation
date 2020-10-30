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
import json
import math
import os
import numpy as np
import tensorflow.compat.v1 as tf




class Features(object):
  """Feature keys."""

  # Waveform(s) of audio observed at receiver(s).
  RECEIVER_AUDIO = 'receiver_audio'

  # Images of each source at each microphone, including reverberation.
  # Images are real valued with shape [sources, microphones, length].
  SOURCE_IMAGES = 'source_images'

  # Boolean diarization labels of shape (sources, length) which indicates
  # whether a source is active or not. For nonexisting source, it is all zeros.
  DIARIZATION_LABELS = 'diarization_labels'

  # Speaker indices (global indices which are contiguous over all training data
  # starting with 0) that are present in this meeting or meeting chunk with
  # shape (sources,). If number of speakers present in the meeting is less
  # than sources, for a non-existing speaker/source, the speaker index is
  # set to -1. Note that, for a meeting sub-block, we still have all the
  # speaker indices in the meeting even if not all the speakers are present
  # in that meeting sub-block.
  SPEAKER_INDEX = 'speaker_indices'


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


def _read_meeting_list(meeting_list, meeting_length_type):
  """Reads meeting list from json file to get necessary information.

  Args:
    meeting_list: A meeting list read from a json file.
    meeting_length_type: One of 'maximum', 'minimum' or 'average'.
      Since typically meeting lengths are not fixed, we can
      set the training/eval length to the maximum, minimum or average meeting
      length in the json file based on the value of this argument. We
      eventually pad or clip individual meetings to attain the desired constant
      meeting length in our data reading pipeline.
  Returns:
   num_meetings: Number of meetings.
   max_num_spk_per_meeting: Maximum number of speakers in a meeting.
   max_num_utt_per_spk: Maximum number of utterances per speaker.
   max_dia_seg_per_utt: Maximum diarization segments per utterance.
   max_utt_length: Maximum utterance length.
   meeting_length: Meeting length that will be used.
   speaker_ids: A list of speaker ids that appear in meetings.
  """
  max_num_spk_per_meeting = 0
  max_num_utt_per_meeting = 0
  meeting_lengths = []
  speaker_id_to_count = collections.defaultdict(int)
  num_meetings = len(meeting_list)
  total_spk = 0
  total_utt = 0
  max_utt_length = 0
  max_num_utt_per_spk = 0
  max_dia_seg_per_utt = 0
  for one_meeting in meeting_list:
    sources_start_end = one_meeting['utterance_start_end']
    meeting_length = int(one_meeting['duration'])
    num_utt_in_meeting = len(sources_start_end)
    max_num_utt_per_meeting = max(max_num_utt_per_meeting, num_utt_in_meeting)
    utt2spk = []
    spk2wavs = collections.defaultdict(list)
    spk_utt_idx = collections.defaultdict(int)
    for start, end, spkid, wav_path in  sources_start_end:
      max_utt_length = max(max_utt_length, end - start)
      utt2spk.append(spkid)
      spk2wavs[spkid].append(wav_path)
      speaker_id_to_count[spkid] += 1
      spk_utt_idx[spkid] += 1
      diarization_info = \
          one_meeting['diarization_label'][spkid][spk_utt_idx[spkid] - 1]
      num_seg_in_utt = len(diarization_info)
      max_dia_seg_per_utt = max(max_dia_seg_per_utt, num_seg_in_utt)
    speakers_in_meeting = list(set(utt2spk))
    num_spk = len(speakers_in_meeting)
    for spkid in speakers_in_meeting:
      max_num_utt_per_spk = max(max_num_utt_per_spk,
                                len(set(spk2wavs[spkid])))
    max_num_spk_per_meeting = max(max_num_spk_per_meeting, num_spk)
    total_spk += num_spk
    total_utt += num_utt_in_meeting
    meeting_lengths.append(meeting_length)
  if meeting_length_type == 'maximum':
    meeting_length = int(math.ceil(np.max(meeting_lengths)))
  elif meeting_length_type == 'minimum':
    meeting_length = int(math.floor(np.min(meeting_lengths)))
  elif meeting_length_type == 'average':
    meeting_length = int(round(np.mean(meeting_lengths)))
  elif isinstance(meeting_length_type, int):
    meeting_length = meeting_length_type
  else:
    raise ValueError(f'Unknown meeting_length_type={meeting_length_type}')
  speaker_ids = sorted(speaker_id_to_count.keys())
  tf.logging.info('Read %s meetings from json file.', num_meetings)
  tf.logging.info('Average number of speakers per meeting = %f.',
                  total_spk / num_meetings)
  tf.logging.info('Average number of utterances per speaker = %f.',
                  total_utt / total_spk)
  return (num_meetings, max_num_spk_per_meeting, max_num_utt_per_spk,
          max_dia_seg_per_utt, max_utt_length,
          meeting_length, speaker_ids)


def _pad_mics_tf(signal, new_mics):
  """Pads new mic channels to an input tensor and returns the updated tensor.

  Args:
    signal: A tf.tensor of shape (input_mics, samples)
    new_mics: The number of new mic channels to be added (integer scalar tensor)
  Returns:
    padded_signal: A tf.tensor of shape (input_mics + new_mics, samples)
  """
  # Take first new_mics channels and shift them by 1 sample.
  new_inputs = tf.roll(signal[:new_mics, :], shift=1, axis=-1)
  # Add noise 1e-3 times the RMS value in the signal.
  noise_scale = 1e-3 * tf.sqrt(tf.reduce_mean(tf.square(new_inputs)))
  new_inputs += noise_scale * tf.random.normal(tf.shape(new_inputs))
  return tf.concat((signal, new_inputs), axis=0)


def json_to_dataset(json_file,
                    batch_size,
                    parallel_readers=tf.data.experimental.AUTOTUNE,
                    randomize_order=False,
                    num_examples=-1,
                    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
                    shuffle_buffer_size=5,
                    repeat=True,
                    num_mics=1,
                    sample_rate=16000,
                    use_relative_path=True,
                    meeting_length_type='maximum',
                    num_meeting_subdivisions=1,
                    sensor_noise_range=(0.0, 0.0)):
  r"""Fetches features from a dictionary and source .wav files.

  Args:
    json_file: A json file containing meeting information.
    batch_size: The number of examples to read.
    parallel_readers: Number of dataset.map operations that should happen in
      parallel.
    randomize_order: Whether to randomly shuffle features.
    num_examples: Limit number of examples to this value.  Unlimited if -1.
    prefetch_buffer_size: How many batches to prefecth.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: If True, repeat the dataset.
    num_mics: The expected number of mics in source wav files.
    sample_rate: Sample rate of wav files read.
    use_relative_path: If True, the path for .wav files is relative to the
      json file, otherwise, the paths are absolute.
    meeting_length_type: 'maximum', 'minimum' or 'average'. Can also specify
      an integer value which is the length in samples, which will be used.
    num_meeting_subdivisions: If > 1, chop the meeting in time into this
      many chunks.
    sensor_noise_range: Range of standard deviation for sensor noise. If
      sensor_noise_range[1] <= 0.0, then no sensor noise is added. Otherwise,
      white Gaussian sensor noise with uniformly random standard deviation
      from the provided range is added as the first reference signal.
  Returns:
    A batch_size number of features constructed from wav files.

  Raises:
    ValueError if max_sources_override is less than assumed max number sources.
  """
  tf.logging.info('Reading %s.', json_file)
  with open(json_file, 'r') as f:
    meeting_list = json.load(f)
  (num_meetings, max_num_spk, max_num_utt_per_spk, max_dia_seg_per_utt,
   max_utt_length, samples, speaker_id_list) = _read_meeting_list(
       meeting_list, meeting_length_type)
  tf.logging.info('Maximum number of speakers per meeting = %s', max_num_spk)
  tf.logging.info('Maximum number of utterances per speaker = %s',
                  max_num_utt_per_spk)
  tf.logging.info('Maximum diarization segments per utterance = %s',
                  max_dia_seg_per_utt)
  tf.logging.info('Maximum utterance length in seconds = %s',
                  max_utt_length/sample_rate)
  tf.logging.info('Used meeting length in seconds = %s', samples/sample_rate)
  tf.logging.info('Number of speakers seen in all meetings = %s',
                  len(speaker_id_list))
  tf.logging.info('Using %s parallel readers.', parallel_readers)
  tf.logging.info('shuffle_buffer=%s, prefetch_buffer=%s, num_mics=%s, '
                  'randomize=%s.', shuffle_buffer_size, prefetch_buffer_size,
                  num_mics, randomize_order)
  if use_relative_path:
    base_path = os.path.dirname(json_file)

  spkid2idx = {key: idx for idx, key in enumerate(speaker_id_list)}

  def utterance_info_generator():
    """Yields utterance informations from each meeting.

    Utterance info is in the form of a 6-tuple:
      wav_path, diarization, spkidx, meeting_scale, start, gain.
    """
    default_diarization = np.zeros((max_dia_seg_per_utt, 2), dtype=np.int32)
    default_utt = ('0', default_diarization, -1, 0.0, 0, 0.0)
    for one_meeting in meeting_list:
      meeting_info = collections.defaultdict(list)
      sources_start_end = one_meeting['utterance_start_end']
      num_utt_in_meeting = len(sources_start_end)
      spk_num_in_meeting = {}
      new_spknum = 0
      spkids_in_meeting = []
      spk_utt_idx = collections.defaultdict(int)
      meeting_scale = float(one_meeting['meeting_scale'])
      for utt_idx in range(num_utt_in_meeting):
        start, end, spkid, wav_path = sources_start_end[utt_idx]
        spkidx = spkid2idx[spkid]
        if start >= samples:
          continue
        if end >= samples:
          end = samples
        if spkidx in spk_num_in_meeting:
          spknum = spk_num_in_meeting[spkidx]
        else:
          spknum = new_spknum
          if spknum > max_num_spk:
            continue
          spkids_in_meeting.append(spkidx)
          spk_num_in_meeting[spkidx] = spknum
          new_spknum += 1
        if use_relative_path:
          wav_path = os.path.join(base_path, wav_path)
        gain = one_meeting['utterance_gain'][utt_idx]
        # Make diarization_labels array.
        diarization = np.zeros((max_dia_seg_per_utt, 2), dtype=np.int32)
        spk_utt_idx[spknum] += 1
        diarization_info = \
            one_meeting['diarization_label'][spkid][spk_utt_idx[spknum] - 1]
        # Go over diarization segments in utterance.
        for i, segment_st_end in enumerate(diarization_info):
          segment_start, segment_end = segment_st_end
          if segment_start >= samples:
            continue
          if segment_end > samples:
            segment_end = samples
          adjusted_start = segment_start - start
          adjusted_end = segment_end - start
          diarization[i, 0] = adjusted_start
          diarization[i, 1] = adjusted_end
        meeting_info[spknum].append((wav_path, diarization, spkidx,
                                     meeting_scale, start, gain))
      for spknum in range(max_num_spk):
        if spknum in meeting_info:
          for utt in range(max_num_utt_per_spk):
            if utt < len(meeting_info[spknum]):
              yield meeting_info[spknum][utt]
            else:
              yield default_utt
        else:
          for utt in range(max_num_utt_per_spk):
            yield default_utt

  utterance_info_list = list(utterance_info_generator())
  # No need for the original meeting_list from now on.
  del meeting_list

  num_utterances = len(utterance_info_list)
  tensor_shape = [(num_utterances, 1),
                  (num_utterances, max_dia_seg_per_utt, 2),
                  (num_utterances, 1),
                  (num_utterances, 1),
                  (num_utterances, 1),
                  (num_utterances, 1)]
  tensor_type = [np.string_, np.int32, np.int32, np.float32,
                 np.int32, np.float32]

  (wav_paths, diarizations, spkindices, meeting_scales, start_samples,
   utterance_gains) = [np.reshape(
       tensor, tensor_shape[i]).astype(tensor_type[i]) for i, tensor in
                       enumerate(list(zip(*utterance_info_list)))]

  dataset = tf.data.Dataset.from_tensor_slices(
      (wav_paths, diarizations, spkindices, meeting_scales, start_samples,
       utterance_gains))
  if repeat:
    dataset = dataset.repeat()

  if randomize_order:
    # Randomize meeting order for each epoch through the dataset.
    dataset = dataset.batch(max_num_spk * max_num_utt_per_spk)
    dataset = dataset.shuffle(num_meetings)
    dataset = dataset.flat_map(
        lambda w, d, s, m, t, u: tf.data.Dataset.from_tensor_slices(
            (w, d, s, m, t, u)))

  # Read in wav files.
  def decode_wav(wav):
    audio_bytes = tf.read_file(wav)
    waveform, _ = tf.audio.decode_wav(audio_bytes,
                                      desired_samples=max_utt_length)
    waveform = tf.transpose(waveform)
    num_read_mics = tf.shape(waveform)[0]
    waveform = tf.cond(num_read_mics >= num_mics,
                       lambda: waveform[:num_mics, :],
                       lambda: _pad_mics_tf(waveform, num_mics - num_read_mics))
    waveform = tf.reshape(waveform, (num_mics, max_utt_length))
    return waveform

  def decode_wav_or_return_zeros(wav, gain=1.0):
    return tf.cond(
        tf.equal(wav, '0'),
        lambda: tf.zeros((num_mics, max_utt_length), dtype=tf.float32),
        lambda: gain * decode_wav(wav))

  def utterance_reader(wav_path, diarization, spkidx, meet_scale, start, gain):
    """Reads wave file for utterance and scale it."""
    utt_tensor = decode_wav_or_return_zeros(wav_path[0], gain=gain)
    return utt_tensor, diarization, spkidx, meet_scale, start

  # Sandwich heavy IO part between prefetch's.
  dataset = dataset.prefetch(parallel_readers)
  dataset = dataset.map(utterance_reader,
                        num_parallel_calls=parallel_readers)
  dataset = dataset.prefetch(parallel_readers)

  def pad_utterance(utt_tensor, diarization, spkidx, meeting_scale, start):
    """Pads utterance to meeting length.

    Args:
      utt_tensor: Utterance with shape (num_mics, max_utt_length).
      diarization: Diarization with shape (max_dia_seg_per_utt, 2).
      spkidx: Speaker index (global) for the utterance.
      meeting_scale: Target meeting scale.
      start: Start index of utterance in the meeting.
    Returns:
      utt_tensor_padded: Padded utt tensor (num_mics, samples + max_utt_length)
      diarization_padded: Diarization updated using the start index.
      spkidx: Speaker index passed unchanged.
      meeting_scale: Target meeting scale passed unchanged.
    """
    start = start[0]
    end_paddings = samples - start
    utt_tensor_padded = tf.pad(utt_tensor, ((0, 0), (start, end_paddings)))
    diarization_padded = start + diarization
    return utt_tensor_padded, diarization_padded, spkidx, meeting_scale

  dataset = dataset.map(pad_utterance,
                        num_parallel_calls=parallel_readers)

  dataset = dataset.batch(max_num_utt_per_spk)

  def make_reference(utt_tensor, diarization, spkidx, meeting_scale):
    """Makes a reference from fixed length utterance tensors.

    Args:
      utt_tensor: Utterances with shape
        (max_num_utt_per_spk, num_mics, samples + max_utt_len)
      diarization: Diarization ranges with shape
        (max_num_utt_per_spk, max_dia_seg_per_utt, 2).
      spkidx: Speaker indices (repeated) with shape (max_num_utt_per_spk)
      meeting_scale: Target meeting scale (repeated).
    Returns:
      reference: Meeting audio with shape (num_mics, samples)
      diarization_labels: tf.bool with shape (samples)
      spkidx: Scalar speaker index.
      meeting_scale: Target meeting scale.
    """
    reference_waveform = tf.reduce_sum(utt_tensor, axis=0)
    reference_waveform = reference_waveform[:, :samples]
    diarization = tf.reshape(diarization,
                             (max_num_utt_per_spk * max_dia_seg_per_utt, 2))
    active_samples_list = [
        tf.range(diarization[i, 0], diarization[i, 1]) for i in
        range(max_num_utt_per_spk * max_dia_seg_per_utt)]
    active_samples = tf.reshape(
        tf.concat(active_samples_list, axis=0), (-1, 1))
    dia_full_init = tf.zeros((samples + max_utt_length, 1), dtype=tf.int32)
    dia_full = tf.tensor_scatter_add(
        dia_full_init, active_samples, tf.ones(tf.shape(active_samples),
                                               dtype=tf.int32))
    dia_full = tf.cast(dia_full[:samples, 0], dtype=tf.bool)
    spkidx = spkidx[0]
    meeting_scale = meeting_scale[0]
    return reference_waveform, dia_full, spkidx, meeting_scale

  dataset = dataset.map(make_reference,
                        num_parallel_calls=parallel_readers)
  dataset = dataset.batch(max_num_spk)

  # If num_meeting_subdivisions > 1, split time-dependent meeting data in time
  # into num_meeting_subdivisions equal chunks. Note that speaker ids and
  # meeting_scale are repeated for each chunk.
  if num_meeting_subdivisions > 1:
    def chop_meeting_data(reference_waveforms, diarization_labels, speaker_ids,
                          meeting_scale, nsplit=num_meeting_subdivisions):
      samples = tf.shape(reference_waveforms)[-1]
      new_samples = nsplit * (samples // nsplit)
      reference_waveforms = tf.stack(
          tf.split(reference_waveforms[..., :new_samples],
                   nsplit, axis=-1), axis=0)
      diarization_labels = tf.stack(
          tf.split(diarization_labels[..., :new_samples],
                   nsplit, axis=-1), axis=0)
      speaker_ids = tf.reshape(speaker_ids, (1, max_num_spk))
      speaker_ids = tf.broadcast_to(speaker_ids, (nsplit, max_num_spk))
      meeting_scale = meeting_scale[0] * tf.ones((nsplit, max_num_spk))
      return tf.data.Dataset.from_tensor_slices((reference_waveforms,
                                                 diarization_labels,
                                                 speaker_ids,
                                                 meeting_scale))
    dataset = dataset.flat_map(chop_meeting_data)
    samples = (samples // num_meeting_subdivisions)

  # Build mixture and sources waveforms.
  def combine_mixture_and_sources(reference_waveforms, diarization_labels,
                                  speaker_ids, meeting_scale):
    # waveforms has shape (num_sources, num_mics, num_samples).
    speaker_ids = tf.reshape(speaker_ids, (max_num_spk,))
    meeting_scale = meeting_scale[0]
    mixture_waveform = tf.reduce_sum(reference_waveforms, axis=0)
    current_mixture_scale = tf.reduce_max(tf.abs(mixture_waveform))
    # Note that when meetings are chopped, we cannot apply a meeting level
    # scale. Instead, we apply the scale in the chunk level so that each
    # chunk has a maximum scale equal to the meeting_scale. However, we should
    # not apply any gain to an all noise chunk to avoid amplifying the noise,
    # so we try not to scale those chunks by checking the current_mixture_scale
    # value.
    scale_refs = tf.cond(current_mixture_scale > 0.005,
                         lambda: meeting_scale / current_mixture_scale,
                         lambda: 1.0)
    reference_waveforms *= scale_refs
    num_sources = max_num_spk
    if sensor_noise_range[1] > 0.0:
      num_sources += 1
      sensor_noise_gain = tf.random.uniform((), minval=sensor_noise_range[0],
                                            maxval=sensor_noise_range[1])
      sensor_noise = sensor_noise_gain * tf.random.normal(
          (1, num_mics, samples))
      reference_waveforms = tf.concat(
          (sensor_noise, reference_waveforms), axis=0)
    mixture_waveform = tf.reduce_sum(reference_waveforms, axis=0)
    reference_waveforms.set_shape((num_sources, num_mics, samples))
    mixture_waveform.set_shape((num_mics, samples))
    diarization_labels.set_shape((max_num_spk, samples))
    speaker_ids.set_shape((max_num_spk,))
    return {'receiver_audio': mixture_waveform,
            'source_images': reference_waveforms,
            'diarization_labels': diarization_labels,
            'speaker_indices': speaker_ids,
            }
  dataset = dataset.map(combine_mixture_and_sources,
                        num_parallel_calls=parallel_readers)
  if randomize_order and num_meeting_subdivisions > 1:
    # It would be good to shuffle examples to avoid having all examples
    # coming from a single meeting when we split a meeting.
    dataset = dataset.shuffle(shuffle_buffer_size * num_meeting_subdivisions)
  dataset = dataset.prefetch(prefetch_buffer_size)
  dataset = dataset.take(num_examples)
  dataset = dataset.batch(batch_size, drop_remainder=True)

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
    json_file = params.get('input_data', None)
    io_params = params.get('io_params', {})
    batch_size = params.get('batch_size', None)
    randomize_order = params.get('randomize_order', False)
    io_params['randomize_order'] = randomize_order
    return json_to_dataset(json_file,
                           batch_size,
                           **io_params)
