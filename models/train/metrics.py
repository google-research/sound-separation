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


def calculate_signal_to_noise_ratio_from_power(
    signal_power, noise_power, epsilon):
  """Computes the signal to noise ratio given signal_power and noise_power.

  Args:
    signal_power: A tensor of unknown shape and arbitrary rank.
    noise_power: A tensor matching the signal tensor.
    epsilon: An optional float for numerical stability, since silences
      can lead to divide-by-zero.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  # Pre-multiplication and change of logarithm base.
  constant = tf.cast(10.0 / tf.log(10.0), signal_power.dtype)

  return constant * tf.log(
      tf.truediv(signal_power + epsilon, noise_power + epsilon))


def calculate_signal_to_noise_ratio(signal, noise, epsilon=1e-8):
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
    return tf.reduce_mean(tf.square(x), reduction_indices=[-1])

  return calculate_signal_to_noise_ratio_from_power(
      power(signal), power(noise), epsilon)


def signal_to_noise_ratio_gain_invariant(estimate, target, epsilon=1e-8):
  """Computes the signal to noise ratio in a gain invariant manner.

  This computes SNR assuming that the signal equals the target multiplied by an
  unknown gain, and that the noise is orthogonal to the target.

  This quantity is also known as SI-SDR [1, equation 5].

  This function estimates SNR using a formula given e.g. in equation 4.38 from
  [2], which gives accurate results on a wide range of inputs, and yields a
  monotonically decreasing value when target or estimate scales toward zero.

  [1] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey,
  "SDR--half-baked or well done?",ICASSP 2019,
  https://arxiv.org/abs/1811.02508.
  [2] Magnus Borga, "Learning Multidimensional Signal Processing"
  https://www.diva-portal.org/smash/get/diva2:302872/FULLTEXT01.pdf

  Args:
    estimate: An estimate of the target of size [..., samples].
    target: A ground truth tensor, matching estimate above.
    epsilon: An optional float introduced for numerical stability in the
      projections only.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  def normalize(x):
    power = tf.reduce_sum(tf.square(x), keepdims=True, reduction_indices=[-1])
    return tf.multiply(x, tf.rsqrt(tf.maximum(power, 1e-16)))

  normalized_estimate = normalize(estimate)
  normalized_target = normalize(target)
  cosine_similarity = tf.reduce_sum(
      tf.multiply(normalized_estimate, normalized_target),
      reduction_indices=[-1])
  squared_cosine_similarity = tf.square(cosine_similarity)
  normalized_signal_power = squared_cosine_similarity
  normalized_noise_power = 1. - squared_cosine_similarity

  # Computing normalized_noise_power as the difference between very close
  # floating-point numbers is not accurate enough for this case, so when
  # normalized_signal power is close to 0., we use an alternate formula.
  # Both formulas are accurate enough at the 'seam' in float32.
  normalized_noise_power_direct = tf.reduce_sum(
      tf.square(normalized_estimate -
                normalized_target * tf.expand_dims(cosine_similarity, -1)),

      reduction_indices=[-1])
  normalized_noise_power = tf.where(
      tf.greater_equal(normalized_noise_power, 0.01),
      normalized_noise_power,
      normalized_noise_power_direct)

  return calculate_signal_to_noise_ratio_from_power(
      normalized_signal_power, normalized_noise_power, epsilon)


def signal_to_noise_ratio_residual(estimate, target, epsilon=1e-8):
  """Computes the signal to noise ratio using residuals.

  This computes the SNR in a "statistical fashion" as the logarithm of the
  relative residuals. The signal is defined as the original target, and the
  noise is the residual between the estimate and the target. This is
  proportional to log(1 - 1/R^2).

  Args:
    estimate: An estimate of the target of size [..., samples].
    target: A ground truth tensor, matching estimate above.
    epsilon: An optional float for numerical stability, since silences
      can lead to divide-by-zero.

  Returns:
    A tensor of size [...] with SNR computed between matching slices of the
    input signal and noise tensors.
  """
  return calculate_signal_to_noise_ratio(target, target - estimate,
                                         epsilon=epsilon)
