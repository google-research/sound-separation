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
"""Mix estimates to match references according to a loss function, [1] eq. (3).

[1] Scott Wisdom, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss, Kevin Wilson,
    John R. Hershey, "Unsupervised Sound Separation Using Mixture Invariant
    Training", https://arxiv.org/abs/2006.12701.

"""

import itertools
from typing import Any, Callable, List, Optional, Tuple
import tensorflow.compat.v1 as tf


def _smart_static_dim(tensor: tf.Tensor, i: int):
  """Static size for dimension `i`."""
  static_shape = tensor.shape
  return (static_shape[i].value if hasattr(static_shape[i], 'value')
          else static_shape[i])


def apply_and_get_mix_matrix(
    loss_fn: Callable[..., tf.Tensor],
    reference: tf.Tensor,
    estimate: tf.Tensor,
    mix_allowlist: Optional[List[Tuple[int, ...]]] = None,
    mix_denylist: Optional[List[Tuple[int, ...]]] = None) -> Any:
  """Computes the loss of best mixing of estimate and returns the mixing matrix.

  Note that loss_fn must in general handle an arbitrary number of sources, since
  this function may expand in that dimension to get losses on all
  reference-estimate pairs.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source ...] tensor
        estimate [batch, source ...] tensor
      Returns
        A [batch, source] tensor of dtype=tf.float32
    reference: [batch, ref_num_sources, ...] tensor.
    estimate: [batch, est_num_sources, ...] tensor.
    mix_allowlist: list of tuples of len est_source, only consider these mixes.
    mix_denylist: list of tuples of len est_source, don't consider these mixes.

  Returns:
    Returns the loss corresponding to the best mixing from the estimates and a
    tensor for the mix_matrix of shape:
    [batch, ref_num_sources, est_num_sources].

  Raises:
    ValueError if ref_source > est_source.
  """
  reference = tf.convert_to_tensor(reference)
  estimate = tf.convert_to_tensor(estimate)
  batch = _smart_static_dim(reference, 0)
  ref_num_sources = _smart_static_dim(reference, 1)
  est_num_sources = _smart_static_dim(estimate, 1)

  if ref_num_sources > est_num_sources:
    raise ValueError(
        'ref_num_sources {} should be <= est_num_sources {}.'.format(
            ref_num_sources, est_num_sources))

  losses = []
  idxs = []
  # This itertools call returns all possible assignments from estimates to
  # references, E.g. itertools.product(range(2), repeat=3) produces:
  # (0, 0, 0)
  # (0, 0, 1)
  # (0, 1, 0)
  # (0, 1, 1)
  # (1, 0, 0)
  # (1, 0, 1)
  # (1, 1, 0)
  # (1, 1, 1)
  for idx in itertools.product(range(ref_num_sources), repeat=est_num_sources):
    if mix_allowlist and idx not in mix_allowlist:
      continue
    if mix_denylist and idx in mix_denylist:
      continue
    mix_matrix = tf.one_hot(tf.constant(idx), ref_num_sources, axis=0,
                            dtype=tf.float32)[tf.newaxis]
    # mix_matrix is shape (1, ref_num_sources, est_num_sources).
    estimate_mixed = tf.matmul(mix_matrix, estimate)
    # estimate_mixed is shape [batch, ref_num_sources, ...].
    losses.append(tf.reduce_mean(loss_fn(reference, estimate_mixed), axis=1,
                                 keepdims=True))
    idxs.append(idx)

  loss_matrix = tf.concat(losses, axis=1)
  # loss_matrix is shape [batch, len(idxs)].
  idx_argmin = tf.argmin(loss_matrix, axis=1)
  idx_argmin = tf.expand_dims(idx_argmin, 1)
  # idx_argmin shape is shape [batch, 1].

  loss_best_mixture = tf.gather_nd(loss_matrix, idx_argmin, batch_dims=1)

  idxs_tf = tf.concat([tf.constant(idx)[tf.newaxis, :] for idx in idxs], axis=0)
  idxs_tf = tf.tile(tf.expand_dims(idxs_tf, 0), [batch, 1, 1])
  # idxs_tf is shape [batch, est_num_sources, len(idxs)].
  idxs_best = tf.gather_nd(idxs_tf, idx_argmin, batch_dims=1)
  # idxs_best is shape [batch, est_num_sources].
  mix_matrix = tf.one_hot(idxs_best, ref_num_sources, axis=1, dtype=tf.float32)

  return loss_best_mixture, mix_matrix


def apply(loss_fn: Callable[..., tf.Tensor],
          reference: tf.Tensor,
          estimate: tf.Tensor,
          mix_allowlist: Optional[List[Tuple[int, ...]]] = None,
          mix_denylist: Optional[List[Tuple[int, ...]]] = None) -> Any:
  """Return loss of best mixing of estimates to match references, [1] eq. (3).

  Uses exhaustive search over shape num_ref x num_est binary mixing matrices.

  Note that loss_fn must in general handle an arbitrary number of sources, since
  this function may expand in that dimension to get losses on all
  reference-estimate pairs.

  Args:
    loss_fn: function with the following signature:
      Args
        reference [batch, source ...] tensor
        estimate [batch, source ...] tensor
      Returns
        A [batch, source] tensor of dtype=tf.float32
    reference: [batch, ref_num_sources, ...] tensor.
    estimate: [batch, est_num_sources, ...] tensor.
    mix_allowlist: list of tuples of len est_source, only consider these mixes.
    mix_denylist: list of tuples of len est_source, don't consider these mixes.

  Returns:
    loss, and a [batch, ref_source] tensor of dtype=tf.float32 that is
    a tensor like reference and is estimated mixed to match reference.

  Raises:
    ValueError if ref_source > est_source.
  """
  loss_best_mixture, mix_matrix = apply_and_get_mix_matrix(
      loss_fn, reference, estimate, mix_allowlist, mix_denylist)
  # mix_matrix is shape [batch, ref_num_sources, est_num_sources].
  estimate_mixed = tf.matmul(mix_matrix, estimate)

  return loss_best_mixture, estimate_mixed
