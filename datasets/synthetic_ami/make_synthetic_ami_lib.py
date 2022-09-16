# Copyright 2022 Google LLC
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
"""Library functions for making synthetic AMI dataset."""
import os
from typing import Any, Mapping, Tuple, Union, Optional

import librosa
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768
MAX_INT16 = 32767
MIN_INT16 = -32768


def read_wav(wav_path: str, start: float, duration: float) -> np.ndarray:
  audio, sr = librosa.load(wav_path, sr=None, mono=False, offset=start,
                           duration=duration)
  if sr != SAMPLE_RATE:
    raise ValueError(f'{wav_path}: expected sr={SAMPLE_RATE}, got {sr}.')
  return audio


def float32_to_int16(x_float32: np.ndarray) -> np.ndarray:
  """Convert float32 data to int16.

  Args:
    x_float32: numpy.ndarray of dtype float32.

  Returns:
    numpy.ndarray of dtype int16.

  Raises:
    ValueError if x_float32 is not dtype float32.
  """
  if x_float32.dtype != np.float32:
    raise ValueError(
        f'Expected input with dtype float32, got {x_float32.dtype}.')
  return np.round(x_float32 * MAX_ABS_INT16).clip(MIN_INT16, MAX_INT16).astype(
      np.int16)


def write_wav(wav_path: str, audio: np.ndarray):
  """Write audio to 16-bit PCM wav.

  Args:
    wav_path: Path to wav file.
    audio: float32 or int16, of shape (samples,) or (channels, samples).
  """
  # If signal is float32, convert to int16.
  if audio.dtype == np.float32:
    audio = float32_to_int16(audio)

  wav_dir = os.path.dirname(wav_path)
  if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

  wavfile.write(wav_path, SAMPLE_RATE, audio.T)


def zero_pad(audio: np.ndarray, shift: float,
             desired_samples: int) -> np.ndarray:
  pad_left = int(round(shift * SAMPLE_RATE))
  pad_right = max(0, desired_samples - pad_left - audio.shape[-1])
  pad = [(0, 0)] * (len(audio.shape) - 1) + [(pad_left, pad_right)]
  return np.pad(audio, pad)[..., :desired_samples]


def _log2(values: tf.Tensor) -> tf.Tensor:
  return tf.math.log(values) / tf.math.log(2.0)


def _smart_dim(tensor: tf.Tensor, i: int) -> Union[tf.Tensor, Tuple[int]]:
  """Static or dynamic size for dimension `i`."""
  static_shape = tensor.shape
  dynamic_shape = tf.shape(tensor)
  return (static_shape[i].value if hasattr(static_shape[i], 'value')
          else static_shape[i]) or dynamic_shape[i]


def _enclosing_power_of_2(input_len: Union[int, tf.Tensor]) -> tf.Tensor:
  """Returns the smallest power of 2 greater than or equal to input_len."""
  n = tf.math.ceil(_log2(tf.cast(input_len, tf.float32) - 0.5))
  return tf.cast(tf.round(tf.pow(2., tf.cast(n, tf.float32))), tf.int32)


def _pad_to_length(a: tf.Tensor, length: Union[int, tf.Tensor]) -> tf.Tensor:
  """Zero pad last dimension of a at the end to achieve length."""
  padding = length - _smart_dim(a, -1)
  padding = tf.expand_dims(tf.expand_dims(padding, 0), 0)  # shape [1, 1]
  padding = tf.pad(padding, [[tf.rank(a) - 1, 0], [1, 0]])  # shape [rank, 2]
  return tf.pad(a, padding)


def calculate_multitap_filter(input_signal: tf.Tensor, target_signal: tf.Tensor,
                              filter_len: int = 10,
                              diagload: float = 0.1) -> tf.Tensor:
  """Calculates the multitap filter that takes input_signal closest to target.

  A filter is found and returned such that ||filt * input - target|| is
  minimized.  Here * is the convolution operation and the norm is the L2 norm.

  Args:
    input_signal: (..., signal_len) the signal to filter.
    target_signal: (..., signal_len) the signal to get close to.
    filter_len: FIR filter length.
    diagload: Diagonal loading to stabilize matrix inversion.
  Returns:
    filt: The FIR filter (..., filter_len) that takes input to target.
  """
  target_signal = tf.convert_to_tensor(target_signal)
  input_signal = tf.convert_to_tensor(input_signal)
  forward_fft = tf.signal.rfft
  inverse_fft = tf.signal.irfft
  input_signal = tf.cast(input_signal, dtype=tf.float32)
  target_signal = tf.cast(target_signal, dtype=tf.float32)

  signal_len = _smart_dim(input_signal, -1)
  result_len = signal_len + filter_len - 1
  fft_len = _enclosing_power_of_2(result_len)

  # Computing correlations thru fft.
  ref_fft = forward_fft(_pad_to_length(input_signal, fft_len))
  est_fft = forward_fft(_pad_to_length(target_signal, fft_len))
  ref_conj_ref = tf.math.conj(ref_fft) * ref_fft
  ref_corr = inverse_fft(ref_conj_ref)
  row_range = tf.range(0, -filter_len, -1)
  row_range = tf.where(row_range < 0, fft_len + row_range, row_range)
  row = tf.gather(ref_corr, row_range, axis=-1)
  column = ref_corr[..., :filter_len]
  vec_for_diagload = tf.cast(diagload * tf.one_hot(0, filter_len), row.dtype)
  row += vec_for_diagload
  column += vec_for_diagload
  toeplitz_matrix = tf.linalg.LinearOperatorToeplitz(
      column, row, is_non_singular=True, is_square=True)
  cross_fft = tf.math.conj(ref_fft) * est_fft
  # Cross correlation.
  cross_corr = inverse_fft(cross_fft)
  rhs_range = tf.range(filter_len)
  rhs = tf.gather(cross_corr, rhs_range, axis=-1)

  dense_toeplitz_matrix = toeplitz_matrix.to_dense()

  filt = tf.linalg.solve(dense_toeplitz_matrix, tf.expand_dims(rhs, -1))[..., 0]
  return filt


def convolve(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """Performs (batched) 1D convolution between two tensors.

  c[..., n] = sum_k a[..., k] b[..., n-k]

  Args:
    a: First tensor (..., seqlen_a).
    b: Second tensor (..., seqlen_b).
  Returns:
    Batched convolution result of shape (..., seqlen_a + seqlen_b - 1).
  """
  if a.dtype.is_floating and b.dtype.is_floating:
    out_dtype = a.dtype
  elif a.dtype.is_complex:
    out_dtype = a.dtype
  elif b.dtype.is_complex:
    out_dtype = b.dtype
  ab_len = _smart_dim(a, -1) + _smart_dim(b, -1) - 1
  fft_len = _enclosing_power_of_2(ab_len)
  afft = tf.signal.fft(tf.cast(_pad_to_length(a, fft_len), tf.complex64))
  bfft = tf.signal.fft(tf.cast(_pad_to_length(b, fft_len), tf.complex64))
  ab = tf.signal.ifft(afft * bfft)
  return tf.cast(ab[..., :ab_len], out_dtype)


def filter_with_best_filter(headset_audio: np.ndarray,
                            micarray_audio: np.ndarray,
                            filter_len: int = 3200) -> np.ndarray:
  """Estimate best LTI FIR filter that maps headset_audio to micarray_audio.

  Args:
    headset_audio: Audio from headset, of shape (samples,).
    micarray_audio: Audio from headset, of shape (samples,).
    filter_len: Length of filter in samples.

  Returns:
    Filtered headset audio of shape (samples,).
  """
  filt = calculate_multitap_filter(
      tf.constant(headset_audio),
      tf.constant(micarray_audio),
      filter_len=filter_len)
  # Pad end of headset_audio before filtering, to get nice reverberant tail.
  filtered_audio = convolve(
      tf.constant(np.pad(headset_audio, (0, filter_len))), filt)
  return filtered_audio.numpy()


def row_to_example_with_filtering(
    row: Mapping[str, Any],
    ami_directory: str,
    mic_name: Optional[str] = 'Array1-01',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Create data for an example from a row of the mixing CSV.

  Args:
    row: row of CSV as dict.
    ami_directory: path of downloaded AMI dataset.
    mic_name: The name of the microphone, e.g., Array1-01, Array1-02 ...

  Returns:
    receiver_audio, (samples,) mixture.
    source_images, (3, samples) reverberant filtered headset source images.
    source_audio, (3, samples) anechoic close-talking headset sources.
  """
  wav_bg = os.path.join(ami_directory, row['wav_bg'])
  seg_start_bg = row['seg_start_bg']
  seg_end_bg = row['seg_end_bg']
  offset_bg = row['offset_bg']
  duration_bg = row['duration_bg']
  wav_fg = os.path.join(ami_directory, row['wav_fg'])
  seg_start_fg = row['seg_start_fg']
  seg_end_fg = row['seg_end_fg']
  offset_fg = row['offset_fg']
  duration_fg = row['duration_fg']
  shift_fg = row['shift_fg']

  # Read the distant mic mixture audio.
  wav_mix_bg = wav_bg.split('.Headset')[0] + f'.{mic_name}.wav'
  mix_bg = read_wav(wav_mix_bg, seg_start_bg, seg_end_bg - seg_start_bg)
  wav_mix_fg = wav_fg.split('.Headset')[0] + f'.{mic_name}.wav'
  mix_fg = read_wav(wav_mix_fg, seg_start_fg, seg_end_fg - seg_start_fg)

  # Read the headset audio.
  headset_bg = read_wav(wav_bg, seg_start_bg, seg_end_bg - seg_start_bg)
  headset_fg = read_wav(wav_fg, seg_start_fg, seg_end_fg - seg_start_fg)

  speech_bg = filter_with_best_filter(
      tf.constant(headset_bg), tf.constant(mix_bg))
  speech_fg = filter_with_best_filter(
      tf.constant(headset_fg), tf.constant(mix_fg))

  # Extract clips from headset and filtered headset.
  offset_bg_samples = int(np.round(offset_bg * SAMPLE_RATE))
  duration_bg_samples = int(np.round(duration_bg * SAMPLE_RATE))
  mix_bg = mix_bg[offset_bg_samples:offset_bg_samples + duration_bg_samples]
  headset_bg = headset_bg[offset_bg_samples:offset_bg_samples +
                          duration_bg_samples]
  speech_bg = speech_bg[offset_bg_samples:offset_bg_samples +
                        duration_bg_samples]

  offset_fg_samples = int(np.round(offset_fg * SAMPLE_RATE))
  duration_fg_samples = int(np.round(duration_fg * SAMPLE_RATE))
  headset_fg = headset_fg[offset_fg_samples:offset_fg_samples +
                          duration_fg_samples + 3200]
  headset_fg = zero_pad(headset_fg, shift_fg,
                        speech_bg.shape[-1])[:duration_bg_samples]
  speech_fg = speech_fg[offset_fg_samples:offset_fg_samples +
                        duration_fg_samples + 3200]
  speech_fg = zero_pad(speech_fg, shift_fg,
                       speech_bg.shape[-1])[:duration_bg_samples]

  # Background noise is residual from bg filtering.
  noise_bg = mix_bg - speech_bg

  source_images = np.stack([noise_bg, speech_bg, speech_fg], axis=0)
  receiver_audio = np.sum(source_images, axis=0)
  source_audio = np.stack([np.zeros_like(headset_bg), headset_bg, headset_fg],
                          axis=0)

  return receiver_audio, source_images, source_audio
