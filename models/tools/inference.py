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
"""Inference for a separation model."""

import distutils.util
import numpy as np
import tensorflow.compat.v1 as tf


def strtobool(v: str) -> bool:
  return bool(distutils.util.strtobool(v))


def _pad_input(input_tensor: tf.Tensor, new_mics: int) -> tf.Tensor:
  """Pads new mic channels to an input tensor and returns the updated tensor.

  Args:
    input_tensor: A tensor of shape (samples, input_mics)
    new_mics: The number of new mic channels to be added (integer)
  Returns:
    padded_tensor: A tensor of shape (samples, input_mics + new_mics)
  """
  # Take first new_mics channels and shift them by 1 sample.
  new_inputs = tf.roll(input_tensor[:, :new_mics], shift=1, axis=0)
  # Add noise 1e-3 times the RMS value in the signal.
  noise_scale = 1e-3 * tf.sqrt(tf.reduce_mean(tf.square(new_inputs)))
  new_inputs += noise_scale * tf.random.normal(tf.shape(new_inputs))
  return tf.concat((input_tensor, new_inputs), axis=-1)


def read_wav_file(input_wav_file, input_channels=None, scale_input=False):
  """Reads a wav file in tensorflow.

  Args:
    input_wav_file: File name.
    input_channels: Number of channels to truncate/extend the input signal to.
    scale_input: If True, scale input signal to have an absolute max of 0.99.
  Returns:
    input_wav: A tensor of type tf.float32 with wav file contents with
      shape [samples, mics].
    sample_rate: A scalar tensor of type tf.int32, sample rate in Hz.
  """
  input_wav = tf.read_file(input_wav_file)
  input_wav, sample_rate = tf.audio.decode_wav(input_wav)
  input_mics = tf.shape(input_wav)[-1]
  # We add additional microphone signals when input_channels is larger
  # than the number of microphones in the input signal. If the input has more
  # microphones, we pick only the desired number of microphones.
  if input_channels > 0:
    input_wav = tf.cond(
        input_mics < input_channels,
        lambda: _pad_input(input_wav, input_channels - input_mics),
        lambda: input_wav[:, :input_channels])
  if scale_input:
    input_wav = 0.99 * input_wav / tf.reduce_max(tf.abs(input_wav))
  return input_wav, sample_rate


def write_wav_file(output_wav_file, output_wav_tensor, sample_rate=16000,
                   output_channels=0, write_outputs_separately=False,
                   num_channels=2, channel_name='source'):
  """Writes a wav file or multiple wav files in tensorflow.

  Args:
    output_wav_file: Output .wav file path, or the path to derive multiple
      output .wav file names from.
    output_wav_tensor: Output signal tensor of shape [samples, channels].
    sample_rate: Sample rate in Hz (int or tensor of type tf.int32).
    output_channels: If positive, truncate output channels to this number.
    write_outputs_separately: If True, write separate outputs for each channel.
    num_channels: Number of channels in the output_wav_tensor.
    channel_name: What to use as the channel descriptive name in individual
      channel outputs when write_outputs_separately=True.
  Returns:
    A list of tf.write_file ops.
  """
  n_outputs_to_write = num_channels
  if output_channels > 0:
    n_outputs_to_write = min(output_channels, n_outputs_to_write)
  write_output_ops = []
  if write_outputs_separately:
    for i in range(n_outputs_to_write):
      outwav_name = output_wav_file.replace('.wav', f'_{channel_name}{i}.wav')
      outwav = tf.audio.encode_wav(output_wav_tensor[:, i:i+1], sample_rate)
      write_output_ops.append(tf.write_file(outwav_name, outwav))
  else:
    outwav = tf.audio.encode_wav(output_wav_tensor[:, :n_outputs_to_write],
                                 sample_rate)
    write_output_ops.append(tf.write_file(output_wav_file, outwav))
  return write_output_ops


class SeparationModel(object):
  """Tensorflow audio separation model."""

  def __init__(self, checkpoint_path, metagraph_path,
               input_tensor_name='input_audio/receiver_audio:0',
               output_tensor_name='denoised_waveforms:0'):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      tf.logging.info('Importing meta graph: %s', metagraph_path)
      new_saver = tf.train.import_meta_graph(metagraph_path)
      print('Restoring model from checkpoint: ', checkpoint_path)
      new_saver.restore(self.sess, checkpoint_path)
    self.input_placeholder = self.graph.get_tensor_by_name(input_tensor_name)
    self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)
    self.input_tensor_name = input_tensor_name
    self.output_tensor_name = output_tensor_name

  def separate(self, mixture_waveform):
    """Separates a mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (batch, num_mics, num_samples),
        or (num_mics, num_samples,) or (num_samples,). Currently only works
        when batch=1 for three dimensional inputs.

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    num_input_dims = np.ndim(mixture_waveform)
    if num_input_dims == 1:
      mixture_waveform_input = mixture_waveform[np.newaxis, np.newaxis, :]
    elif num_input_dims == 2:
      mixture_waveform_input = mixture_waveform[np.newaxis, :]
    elif num_input_dims == 3:
      assert np.shape(mixture_waveform)[0] == 1
      mixture_waveform_input = mixture_waveform
    else:
      raise ValueError('Unsupported number of mixture waveform input '
                       f'dimensions {num_input_dims}.')

    separated_waveforms = self.sess.run(
        self.output_tensor,
        feed_dict={self.input_placeholder: mixture_waveform_input})[0]
    return separated_waveforms


def sqrt_hann_window(length, dtype):
  return tf.sqrt(tf.signal.hann_window(length, dtype=dtype, periodic=True))


class OracleBinaryMasking(object):
  """Oracle binary masking with STFT, implemented in tensorflow."""

  def __init__(self, ws=0.032, hs=0.008, sr=16000.0):
    self.stft_win = int(np.round(ws * sr))
    self.stft_hop = int(np.round(hs * sr))
    self.fft_length = int(2**np.ceil(np.log2(self.stft_win)))

  def _pad_beginning(self, waveform):
    pad_len = int(self.stft_win - self.stft_hop)
    pad_spec = [(0, 0)] * (len(waveform.shape) - 1) + [(pad_len, 0)]
    return tf.pad(waveform, pad_spec)

  def _clip_beginning(self, waveform):
    clip = int(self.stft_win - self.stft_hop)
    return waveform[..., clip:]

  def _stft_forward(self, inp):
    waveform = self._pad_beginning(inp)
    return tf.signal.stft(
        waveform, self.stft_win, self.stft_hop,
        fft_length=self.fft_length,
        window_fn=sqrt_hann_window,
        pad_end=True)

  def _stft_inverse(self, inp):
    waveform = tf.signal.inverse_stft(
        inp, self.stft_win, self.stft_hop,
        fft_length=self.fft_length,
        window_fn=tf.signal.inverse_stft_window_fn(
            self.stft_hop, forward_window_fn=sqrt_hann_window))
    return self._clip_beginning(waveform)

  def _oracle_binary_mask(self, amplitudes_sources):
    amplitudes_max = tf.reduce_max(amplitudes_sources, axis=0, keepdims=True)
    return tf.cast(tf.equal(amplitudes_sources, amplitudes_max),
                   amplitudes_sources.dtype)

  def separate(self, mixture_waveform, source_waveforms):
    """Separates a mixture with oracle binary mask computed from references.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).
      source_waveforms: numpy.ndarray of shape (num_sources, num_samples).

    Returns:
      numpy.ndarray of (num_sources, num_samples) of source estimates.
    """
    stft_sources = self._stft_forward(tf.convert_to_tensor(source_waveforms))
    mask = self._oracle_binary_mask(tf.abs(stft_sources))

    stft_mixture = self._stft_forward(tf.convert_to_tensor(mixture_waveform))
    mask = tf.cast(mask, stft_mixture.dtype)
    separated_waveforms = self._stft_inverse(mask * stft_mixture)
    return separated_waveforms[:, :mixture_waveform.shape[-1]]
