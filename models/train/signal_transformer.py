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
"""Defines SignalTransformer class for converting among signal representations.

stft:
 - (batch,          time) waveform => (batch,          frame, bin) spectrogram
 - (batch, channel, time) waveform => (batch, channel, frame, bin) spectrogram

inverse_stft:
 - (batch,          frame, bin) spectrogram => (batch,          time) waveform
 - (batch, channel, frame, bin) spectrogram => (batch, channel, time) waveform
"""

import immutabledict
import numpy as np
import tensorflow.compat.v1 as tf

from . import signal_util


def sqrt_hann_window(window_length, dtype):
  """Square-root Hann window as a Tensor."""
  return tf.sqrt(tf.signal.hann_window(window_length, dtype=dtype,
                                       periodic=True))


_WINDOW_FN = immutabledict.immutabledict({
    'sqrt_hann': sqrt_hann_window,
    'hann': tf.signal.hann_window,
    'hamming': tf.signal.hamming_window,
    'kaiser': tf.signal.kaiser_window,
})


class SignalTransformer(object):
  """SignalTransformer converts among signal representations.

  From a complex spectrogram, SignalTransformer can compute other
  representations (e.g., various kinds of spectrograms).
  """

  def __init__(self,
               sample_rate,
               window_time_seconds=0.025,
               hop_time_seconds=0.01,
               magnitude_offset=1e-8,
               zeropad_beginning=False,
               num_basis=-1,
               window_fn_name='sqrt_hann'):
    assert magnitude_offset >= 0, 'magnitude_offset must be nonnegative.'

    self.sample_rate = sample_rate
    self.magnitude_offset = magnitude_offset
    self.zeropad_beginning = zeropad_beginning

    # Compute derivative parameters.
    self.samples_per_window = int(round(sample_rate * window_time_seconds))
    self.hop_time_samples = int(round(self.sample_rate * hop_time_seconds))

    if num_basis <= 0:
      self.fft_len = signal_util.enclosing_power_of_two(self.samples_per_window)
    else:
      assert num_basis >= self.samples_per_window
      self.fft_len = num_basis
    self.fft_bins = int(self.fft_len / 2 + 1)
    self.forward_window_fn = _WINDOW_FN[window_fn_name]

  def pad_beginning(self, waveform):
    pad_len = int(self.samples_per_window - self.hop_time_samples)
    pad_spec = [(0, 0)] * (len(waveform.shape) - 1) + [(pad_len, 0)]
    return tf.pad(waveform, pad_spec)

  def clip_beginning(self, waveform):
    clip = int(self.samples_per_window - self.hop_time_samples)
    return waveform[..., clip:]

  def forward(self, waveform):
    return self._stft(waveform)

  def inverse(self, spectrogram):
    return self._inverse_stft(spectrogram)

  def _stft(self, waveform):
    """Compute forward STFT with tf.signal, with optional padding on ends."""
    if self.zeropad_beginning:
      waveform = self.pad_beginning(waveform)
    return tf.signal.stft(
        waveform,
        np.int32(self.samples_per_window),
        np.int32(self.hop_time_samples),
        self.fft_len,
        window_fn=self.forward_window_fn,
        pad_end=True,
        name='complex_spectrogram')

  def _inverse_stft(self, complex_spectrogram):
    """Compute inverse STFT with tf.signal, with optional padding on ends."""
    waveform = tf.signal.inverse_stft(
        complex_spectrogram,
        self.samples_per_window,
        self.hop_time_samples,
        self.fft_len,
        window_fn=tf.signal.inverse_stft_window_fn(
            self.hop_time_samples, forward_window_fn=self.forward_window_fn))
    if self.zeropad_beginning:
      waveform = self.clip_beginning(waveform)
    return waveform
