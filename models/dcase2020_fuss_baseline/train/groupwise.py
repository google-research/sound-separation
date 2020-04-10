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
"""Apply per-source-type loss functions."""

import typing

import tensorflow.compat.v1 as tf

from . import permutation_invariant


def apply(loss_fns: typing.Dict[str, typing.Callable[..., typing.Any]],
          signal_names: typing.List[str],
          reference: tf.Tensor,
          estimate: tf.Tensor,
          permutation_invariant_losses: typing.List[str]):
  """Apply loss functions to the corresponding references and estimates.

  For each kind of signal, gather corresponding references and estimates, and
  apply the loss function.  Scatter-add the results into the loss.

  For elements of signals_names not in loss_fns, no loss will be applied.

  Args:
    loss_fns: dictionary of string -> loss_fn.
      Each string is a name to match elements of signal_names.
      Each loss_fn has the following signature:
      Args
        reference [batch, grouped_source, ...] tensor
        estimate [batch, grouped_source, ...] tensor
      Returns
        A [batch, grouped_source] tensor of dtype=tf.float32
    signal_names: list of names of each signal.
    reference: [batch, source, ...] tensor.
    estimate: [batch, source, ...] tensor.
    permutation_invariant_losses: List of losses to be permutation invariant.

  Returns:
    loss, A [batch, source] tensor of dtype=tf.float32
  """
  if reference.shape[:2] != estimate.shape[:2]:
    raise ValueError('First two axes (batch, source) of reference and estimate'
                     'must be equal, got {}, {}'.format(
                         reference.shape[:2], estimate.shape[:2]))

  batch = reference.shape[0]

  loss = tf.zeros(shape=reference.shape[:2], dtype=tf.float32)
  permuted_estimates = tf.zeros_like(reference)

  # For each kind of signal, e.g. 'speech', 'noise', gather subsets of reference
  # and estimate, apply loss function and scatter-add into the loss tensor.
  for name, loss_fn in loss_fns.items():
    idxs = [idx for idx, value in enumerate(signal_names) if value == name]
    idxs_0 = tf.tile(
        tf.expand_dims(tf.range(batch), 1),
        [1, len(idxs)])
    idxs_1 = tf.tile(
        tf.expand_dims(tf.constant(idxs, dtype=tf.int32), 0),
        [batch, 1])
    idxs_nd = tf.stack([idxs_0, idxs_1], axis=2)
    reference_key = tf.gather_nd(reference, idxs_nd)
    estimate_key = tf.gather_nd(estimate, idxs_nd)

    loss_fn = permutation_invariant.wrap(
        loss_fn,
        enable=name in permutation_invariant_losses)
    loss_key, permuted_estimates_key = loss_fn(reference_key, estimate_key)

    loss = tf.tensor_scatter_add(loss, idxs_nd, loss_key)
    permuted_estimates = tf.tensor_scatter_add(
        permuted_estimates, idxs_nd, permuted_estimates_key)

  return loss, permuted_estimates
