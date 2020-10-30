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
"""Utilities that relate to mixture consistency of separation problems."""

import tensorflow.compat.v1 as tf


def enforce_mixture_consistency_time_domain(mixture_waveforms,
                                            separated_waveforms,
                                            mix_weights=None,
                                            mix_weights_type=''):
  """Projection implementing mixture consistency in time domain.

  This projection makes the sum across sources of separated_waveforms equal
  mixture_waveforms and minimizes the unweighted mean-squared error between the
  sum across sources of separated_waveforms and mixture_waveforms. See
  https://arxiv.org/abs/1811.08521 for the derivation.

  Args:
    mixture_waveforms: Tensor of mixture waveforms in waveform format.
    separated_waveforms: Tensor of separated waveforms in source image format.
    mix_weights: None or Tensor of weights used for mixture consistency, shape
        should broadcast with denoised_waveforms. Overrides mix_weights_type.
    mix_weights_type: Type of weights used for mixture consistency. Options are:
        `` - No weighting.
        `magsq` - Mix weights are magnitude-squared of the separated signal.

  Returns:
    Projected separated_waveforms as a Tensor in source image format.
  """
  # Modify the source estimates such that they sum up to the mixture, where
  # the mixture is defined as the sum across sources of the true source
  # targets. Uses the least-squares solution under the constraint that the
  # resulting source estimates add up to the mixture.
  num_sources = tf.shape(separated_waveforms)[1]

  # Add a sources axis to mixture_spectrograms.
  mix = tf.expand_dims(mixture_waveforms, axis=1)
  # mix is now of shape:
  # (batch_size, 1, num_mics, samples).
  mix_estimate = tf.reduce_sum(separated_waveforms, axis=1, keepdims=True)
  # mix_estimate is of shape:
  # (batch_size, 1, num_mics, samples).
  if mix_weights is None:
    if mix_weights_type == 'magsq':
      mix_weights = 1e-8 + tf.reduce_mean(
          tf.square(separated_waveforms), axis=[2, 3], keepdims=True)
      mix_weights /= tf.reduce_sum(mix_weights, axis=1, keepdims=True)
    else:
      mix_weights = (1.0 / num_sources)
  mix_weights = tf.cast(mix_weights, mix.dtype)
  correction = mix_weights * (mix - mix_estimate)
  separated_waveforms = separated_waveforms + correction

  return separated_waveforms
