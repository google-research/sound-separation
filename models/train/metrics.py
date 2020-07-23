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
"""Metrics for source separation."""

import tensorflow.compat.v1 as tf


def calculate_signal_to_noise_ratio(signal, noise, epsilon=1.0e-5):
  """Computes the signal to noise ratio given signal and noise.

  Args:
    signal: A [..., samples] tensor of unknown shape and arbitrary rank.
    noise: A tensor matching the signal tensor.
    epsilon: An optional float for numerical stability, since silences
      can lead to divide-by-zero.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  def power(x):
    return tf.reduce_sum(tf.square(x), reduction_indices=[-1])

  # Pre-multiplication and change of logarithm base.
  constant = tf.cast(10.0 / tf.log(10.0), signal.dtype)

  return constant * tf.log(
      tf.truediv(power(signal) + epsilon, power(noise) + epsilon))


def signal_to_noise_ratio_gain_invariant(estimate, target, epsilon=1.0e-5):
  """Computes the signal to noise ratio in a gain invariant manner.

  This computes SNR in a scale-free manner by projecting the estimate onto the
  target for the signal, and the projection onto the orthogonal subspace for the
  noise.

  Args:
    estimate: An estimate of the target of size [..., samples].
    target: A ground truth tensor, matching estimate above.
    epsilon: An optional float introduced for numerical stability in the
      projections only.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  scaling_factors = tf.rsqrt(
      tf.reduce_sum(tf.square(target), keep_dims=True, reduction_indices=[-1]) +
      epsilon**2.0)
  scaled_target = tf.multiply(target, scaling_factors)
  signal = tf.reduce_sum(
      tf.multiply(estimate, scaled_target),
      keep_dims=True,
      reduction_indices=[-1]) * scaled_target
  noise = estimate - signal

  return calculate_signal_to_noise_ratio(signal, noise)


def signal_to_noise_ratio_residual(estimate, target):
  """Computes the signal to noise ratio using residuals.

  This computes the SNR in a "statistical fashion" as the logarithm of the
  relative residuals. The signal is defined as the original target, and the
  noise is the residual between the estimate and the target. This is
  proportional to log(1 - 1/R^2).

  Args:
    estimate: An estimate of the target of size [..., samples].
    target: A ground truth tensor, matching estimate above.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  return calculate_signal_to_noise_ratio(target, target - estimate)
