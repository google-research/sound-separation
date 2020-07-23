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
r"""Utilities to create summaries in various environments.

Functions in this module assume dictionaries such as the following that define
the tensors to be summarized.

```
import summary_util

scalars = {
    'snr': tf.ones([]),
    ...
}
audio = {
    'waveform': tf.ones([2, 48000, 1]),  # Two mono waveforms.
    ...
}
images = {
    'spectrogram': tf.ones([2, 100, 100, 3])  # Two 100x100 RGB images.
    ...
}
```

Example, compatible with using tf.estimator.Estimator:

```
summary_util.create_summaries(scalars, audio, images)
```
"""

import tensorflow.compat.v1 as tf


def create_summaries(scalars=None,
                     audio=None,
                     images=None,
                     sample_rate=16000,
                     max_audio_outputs=3,
                     max_image_outputs=3):
  """Create training summaries for the given dictionaries of values.

  Args:
    scalars: Dict name -> scalar in form expected by tf.summary.scalar.
    audio: Dict name -> audio, in form expected by tf.summary.audio.
    images: Dict name -> image in form expected by tf.summary.image.
    sample_rate: Audio sample rate.
    max_audio_outputs: Maximum outputs for audio summaries.
    max_image_outputs: Maximum outputs for image summaries.
  """
  scalars = scalars or {}
  audio = audio or {}
  images = images or {}

  for name, tensor in scalars.items():
    tf.summary.scalar(name, tensor)

  for name, tensor in audio.items():
    tf.summary.audio(
        name=name,
        tensor=tensor,
        sample_rate=sample_rate,
        max_outputs=max_audio_outputs)

  for name, tensor in images.items():
    tf.summary.image(
        name=name,
        tensor=tensor,
        max_outputs=max_image_outputs)


def metrics_fn(**scalars):
  return {name: tf.metrics.mean(s) for name, s in scalars.items()}
