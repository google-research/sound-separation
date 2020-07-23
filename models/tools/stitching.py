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
"""Tools for stitching outputs of separation models across short blocks.

To run separation models for long signals, we can run separation for
blocks and combine the results of each block with the other blocks.
For permutation invariant models, we need to resolve the permutation across
blocks. We call this process stitching.
"""

import itertools
from typing import Callable
import tensorflow.compat.v1 as tf

LossType = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def _resolve_permutation(loss_matrix: tf.Tensor) -> tf.Tensor:
  """Resolves permutation from an all-pairs loss_matrix input.

  Args:
    loss_matrix: tensor of shape [batch, source, source]
      axis 1 refers to the reference.
      axis 2 refers to the estimate.
  Returns:
    permutation: tensor of shape [batch, source, 1] such that
      tf.gather_nd(estimates, permutation, 1) returns the permuted estimates
      that achieves the lowest loss.
  """
  batch = tf.shape(loss_matrix)[0]
  source = loss_matrix.shape[1]

  # Compute permutations as vectors of indices into flattened loss matrix.
  # permutations will have shape [batch, source!, source, 1].
  permutations = tf.constant(list(itertools.permutations(range(source))))
  permutations = tf.expand_dims(tf.expand_dims(permutations, 0), 3)
  permutations = tf.tile(permutations, [batch, 1, 1, 1])

  # Expand loss dimensions for gather.
  # loss_matrix.shape will be (batch, source!, source, source)
  loss_matrix = tf.expand_dims(loss_matrix, 1)
  loss_matrix = tf.tile(loss_matrix, [1, permutations.shape[1], 1, 1])

  # Compute the total loss for each permutation.
  # permuted_loss.shape will be (batch, source!)
  permuted_loss = tf.gather_nd(loss_matrix, permutations, batch_dims=3)
  permuted_loss = tf.reduce_sum(permuted_loss, axis=2)

  # Get and return the permutation with the lowest total loss.
  # loss_argmin.shape will be (batch, 1)
  loss_argmin = tf.argmin(permuted_loss, axis=1)
  loss_argmin = tf.expand_dims(loss_argmin, 1)

  # permutation.shape will be (batch, source, 1)
  permutation = tf.gather_nd(permutations, loss_argmin, batch_dims=1)

  return permutation


def _mse_loss(reference: tf.Tensor, estimate: tf.Tensor) -> tf.Tensor:
  """Mean squared error loss between tensors."""
  return tf.reduce_mean(tf.square(reference - estimate), axis=-1)


def _mag_stft_mse_loss(reference: tf.Tensor, estimate: tf.Tensor,
                       frame_length: int = 2048) -> tf.Tensor:
  """Mean squared error loss between magnitude STFT of tensors."""
  ref_stft = tf.signal.stft(reference, frame_length, frame_length // 2,
                            pad_end=True)
  est_stft = tf.signal.stft(estimate, frame_length, frame_length // 2,
                            pad_end=True)
  ref_mag_stft = tf.abs(ref_stft)
  est_mag_stft = tf.abs(est_stft)
  return tf.reduce_mean(tf.square(ref_mag_stft - est_mag_stft), axis=[-1, -2])


def get_window(wintype: str, block_samples: int) -> tf.Tensor:
  """Returns a window tensor of length block_samples.

  Args:
    wintype: Window type as a string.
    block_samples: Number of samples in the window.
  Returns:
    window: A tensor of length block_samples.
  Raises:
    ValueError: When wintype is unknown.
  """
  if wintype == 'vorbis':
    window = tf.signal.vorbis_window(block_samples, dtype=tf.float32)
  elif wintype == 'kaiser-bessel-derived':
    window = tf.signal.kaiser_bessel_derived_window(block_samples,
                                                    dtype=tf.float32)
  elif wintype == 'rectangular':
    window = 1.0/tf.sqrt(2.0) * tf.ones((block_samples,))
  else:
    raise ValueError(f'Window type {wintype} unknown.')
  return window


def sequentially_resolve_permutation(
    blocks: tf.Tensor,
    window: tf.Tensor,
    loss_fn: LossType = _mag_stft_mse_loss) -> tf.Tensor:
  """Resolves permutation between overlapping blocks.

  Args:
    blocks: Waveform in blocks (blocks, sources, samples)
    window: Window function used to obtain the blocks.
    loss_fn: A loss function operating on two tensors and returning
      a loss tensor, reducing loss value over the last dimension.
  Returns:
    perm_blocks: Permute sources to be consistent across blocks with shape
      (blocks, sources, samples)
  """
  num_blocks = tf.shape(blocks)[0]
  num_sources = blocks.shape[1]
  block_samples = blocks.shape[-1]
  hop_samples = block_samples // 2
  # If the blocks were obtained after windowing, we need to compensate
  # for the effect of the window. We can do this by dividing with the window
  # however that can be unstable, so we compensate by rolling the window and
  # multiplying the blocks with the rolled window which ends up having the
  # same windowing effect on begin and end parts of the block. This is
  # numerically more stable and does not give large weight to samples at the
  # edges which can cause problems in practice.
  rolled_window = tf.roll(window, shift=hop_samples, axis=0)
  reweighted_blocks = rolled_window * blocks
  # Split samples dimension into two: begin-part and end-part.
  begin_parts, end_parts = tf.split(reweighted_blocks, 2, axis=-1)

  # We assume axis=1 is the reference, axis=2 is the estimate.
  # End parts of previous frame is the reference, begin parts of current
  # frame is the estimate.
  begin_parts = tf.expand_dims(begin_parts, axis=1)
  end_parts = tf.expand_dims(end_parts, axis=2)
  loss_matrix = loss_fn(end_parts[:-1], begin_parts[1:])
  # loss_matrix has shape (blocks - 1, sources, sources).
  # We find the permutations that aligns each current block with its previous
  # block using batch mode permutation resolution where the blocks
  # dimension is considered as the batch dimension.
  permutation = _resolve_permutation(loss_matrix)
  # permutation has shape (blocks - 1, sources, 1), so we reshape.
  permutation = tf.reshape(permutation, (num_blocks - 1, num_sources))
  # Sequentially update permutations to consider the first block as the
  # reference instead of the previous block to align them all.
  # We do this by sequentially realigning each block with the previous block.
  # Since the previous block is already updated to align with the first block,
  # due to the iterative induction here, we get all aligned signals.
  updated_perm_init = tf.TensorArray(tf.int32, size=num_blocks,
                                     element_shape=(num_sources,))
  previous_perm = tf.range(num_sources)
  # First block of updated perm is identity permutation.
  updated_perm_init = updated_perm_init.write(0, previous_perm)
  def update_perms(i, previous_perm, updated_perm):
    # Note: permutation is missing first block, so we access (i-1)th entry.
    perm_now = tf.gather(permutation[i - 1], previous_perm, axis=0)
    updated_perm = updated_perm.write(i, perm_now)
    previous_perm = perm_now
    i = i + 1
    return i, previous_perm, updated_perm
  while_cond = lambda i, prev_perm, updated_perm: tf.less(i, num_blocks)
  _, _, updated_perm = tf.while_loop(while_cond, update_perms,
                                     [1, previous_perm, updated_perm_init],
                                     back_prop=False, parallel_iterations=1)
  updated_perm = updated_perm.stack()
  # u_perm has shape (n_blocks, n_sources).
  # Obtain the indices for gather_nd.
  batch_index = tf.tile(tf.expand_dims(tf.range(num_blocks), axis=1),
                        [1, num_sources])
  batch_index = tf.reshape(batch_index, [num_blocks, num_sources])
  perm_index = tf.stack([batch_index, updated_perm], axis=2)
  # Now, permute blocks tensor to align all blocks.
  perm_blocks = tf.gather_nd(blocks, perm_index)
  return perm_blocks

