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
"""Functions for spatial filtering, a.k.a beamforming."""

import dataclasses
from typing import Optional, Tuple
import tensorflow.compat.v1 as tf

from . import filtering_projections
from . import shaper
from . import signal_transformer
from . import signal_util


def _smart_shape(t):
  return [d or tf.shape(t)[i] for i, d in enumerate(t.shape.as_list())]


def _complex_to_realimag(matrix: tf.Tensor) -> tf.Tensor:
  matrix_real = tf.real(matrix)
  matrix_imag = tf.imag(matrix)
  return tf.concat([
      tf.concat([matrix_real, -matrix_imag], axis=-1),
      tf.concat([matrix_imag, matrix_real], axis=-1)
  ], axis=-2)


def _hermitian_matrix_solve(matrix: tf.Tensor, rhs: tf.Tensor,
                            method='default') -> tf.Tensor:
  """Matrix_solve using various methods."""
  if method == 'cholesky':
    if matrix.dtype == tf.float32:
      return tf.cholesky_solve(tf.cholesky(matrix), rhs)
    else:
      matrix_realimag = _complex_to_realimag(matrix)
      n = matrix.shape[-1]
      rhs_realimag = tf.concat([tf.real(rhs), tf.imag(rhs)], axis=-2)
      lhs_realimag = tf.cholesky_solve(tf.cholesky(matrix_realimag),
                                       rhs_realimag)
      return tf.complex(lhs_realimag[..., :n, :], lhs_realimag[..., n:, :])
  elif method == 'ls':
    return tf.matrix_solve_ls(matrix, rhs)
  elif method == 'default':
    return tf.matrix_solve(matrix, rhs)
  else:
    raise ValueError(f'Unknown matrix solve method {method}.')


def _add_diagonal_matrix(ryy: tf.Tensor, diagload: float = 1e-3,
                         epsilon: float = 1e-8,
                         use_diagonal_of: Optional[tf.Tensor] = None,
                         ) -> tf.Tensor:
  """Regularize matrix usually before taking its inverse.

  Update ryy matrix with ryy += diagload * diag(matrix) + epsilon * I
  where matrix is either equal to ryy or another matrix given by
  use_diagonal_of parameter and I is the identity matrix and diag(.) is the
  diagonal matrix obtained from its argument.

  Args:
    ryy: A [..., mic, mic] complex64/float32 tensor, covariance matrix.
    diagload: A float32 value.
    epsilon: A float32 value.
    use_diagonal_of: None or another tensor [..., mic, mic] whose diagonal
      is used. If None, diagonal of ryy is used.

  Returns:
    [..., mic, mic] tensor, ryy + diagload * diag(use_diagonal_of) + epsilon*I.
  """

  mic = signal_util.static_or_dynamic_dim_size(ryy, -1)
  if use_diagonal_of is None:
    use_diagonal_of = ryy

  # The eps_identity cannot not be too small and too large relative to ryy, it
  # also needs to deal with the case when ryy is zero. Note that 1e-8 is the
  # default lower bound on signal variance and per T-F unit power.
  diagonal_matrix = (diagload * use_diagonal_of + epsilon) * tf.eye(
      mic, dtype=ryy.dtype)
  return ryy + diagonal_matrix


def _get_beamformer_from_covariances(y_cov: tf.Tensor,
                                     t_cov: tf.Tensor,
                                     diagload: float = 1e-3,
                                     epsilon: float = 1e-8,
                                     refmic: int = 0,
                                     beamformer_type: str = 'wiener'
                                     ) -> tf.Tensor:
  """Calculates beamformers from full covariance estimates.

  Typically mixture signal covariance is estimated from the mixture signal and
  the target covariance is estimated using a mask-based covariance estimation.

  Args:
    y_cov: Mixture signal covariance of shape [..., mic, mic].
    t_cov: Source signal covariance estimate of shape [..., mic, mic, source].
    diagload: diagonal loading factor.
    epsilon: data-independent stabilizer for diagonal loading.
    refmic: Reference mic.
    beamformer_type: 'wiener' or 'mvdr' or 'mpdr'.
  Returns:
    beamformers w of shape [..., mic, source].
  """
  y_cov_rank = tf.get_static_value(tf.rank(y_cov))
  # Last two dimensions in y_cov are (mic, mic) and the rest are variable.
  start = y_cov_rank - 2
  prefix = list(range(start))
  if y_cov_rank < 2:
    raise ValueError('Unsupported y_cov rank {}'.format(y_cov_rank))
  if beamformer_type == 'wiener':
    # Multi-channel wiener filter calculates the beamformer using the
    # formula \phi_{yy}^{-1} \phi_{xx} e_{ref} where \phi_{yy} is the
    # spatial covariance matrix for the mixture signal and \phi_{xx} is the
    # same for the target source signal. e_{ref} is a unit vector with a
    # 1 in the location of the reference mic. We calculate this using
    # matrix_solve where the right hand side includes column vectors for each
    # source, so that we find the beamformers for all sources at once.
    w = _hermitian_matrix_solve(_add_diagonal_matrix(y_cov, diagload, epsilon),
                                t_cov[..., refmic, :])
  elif beamformer_type.startswith('mvdr'):
    # We use mu=0.0. In theory mu=1.0 is equivalent to MCWF but in practice
    # seems to be worse in terms of SNR.
    mu = 0.0
    # Our version of MVDR beamformer is obtained by the formula:
    # (1.0/trace(\phi_{nn}^{-1}) \phi_xx) \phi_{nn}^{-1} \phi_xx e_{ref}
    # where \phi_{nn} is the spatial covariance for all sources except the
    # target source and \phi{xx} is the spatial covariance of the target source.
    # Since we need the trace of the matrix for the denominator, we find the
    # \phi_{nn}^{-1}) \phi_xx matrix first using matrix_solve separately
    # for each source since each source has a different \phi_{nn} unlike MCWF.
    # Then we obtain the trace and multiply the matrix with the e_{ref} which
    # corresponds to picking the `ref` column from the matrix.
    # Since we need to do a different matrix_solve for each source,
    # we need to include the source dimension into batch dimensions unlike MCWF.
    t_cov = tf.transpose(t_cov, prefix + [start + 2, start, start + 1])
    # t_cov has shape [..., source, mic, mic].
    # Find non-target covariance for each source.
    nt_cov = tf.reduce_sum(t_cov, axis=-3, keepdims=True) - t_cov
    y_cov = tf.expand_dims(y_cov, axis=-3)
    nt_inv_t_matrix = _hermitian_matrix_solve(
        _add_diagonal_matrix(nt_cov, diagload=1e-2, epsilon=epsilon,
                             use_diagonal_of=y_cov),
        t_cov)
    scale = tf.reciprocal(mu + tf.linalg.trace(nt_inv_t_matrix) + 1e-8)
    scale = tf.expand_dims(scale, -1)
    # nt_inv_t_matrix has shape [..., source, mic, mic].
    w = scale * nt_inv_t_matrix[..., refmic]
    # w has shape [..., source, mic].
    w = tf.transpose(w, prefix + [start + 1, start])
    # w has shape [..., mic, source].
  elif beamformer_type == 'mpdr':
    # MPDR beamformer is obtained by the formula:
    # (1.0/trace(\phi_{yy}^{-1}) \phi_xx) \phi_{yy}^{-1} \phi_xx e_{ref}
    # where \phi_{yy} is the spatial covariance of the mixture and
    # \phi{xx} is the spatial covariance of the target source.
    # Since we need the trace of the matrix for the denominator, we find the
    # \phi_{yy}^{-1}) \phi_xx matrix first.
    # Then we obtain the trace and multiply the matrix with the e_{ref} which
    # corresponds to picking the `ref` column from the matrix.
    # Since we need to do a different matrix_solve for each source,
    # we need to include the source dimension into batch dimensions unlike MCWF.
    t_cov = tf.transpose(t_cov, prefix + [start + 2, start, start + 1])
    # t_cov has shape [..., source, mic, mic].
    y_cov = tf.expand_dims(y_cov, axis=-3)
    y_cov = tf.broadcast_to(y_cov, tf.shape(t_cov))
    # This finds the whole (mic x mic) matrix of phi_yy^{-1} phi_xx for
    # each (..., source) where x is the target and y is the mixed.
    y_inv_t_matrix = _hermitian_matrix_solve(
        _add_diagonal_matrix(y_cov, diagload, epsilon), t_cov)
    scale = tf.reciprocal(tf.linalg.trace(y_inv_t_matrix) + 1e-8)
    scale = tf.cast(tf.expand_dims(scale, -1), dtype=y_cov.dtype)
    # nt_inv_t_matrix has shape [..., source, mic, mic].
    w = scale * y_inv_t_matrix[..., refmic]
    # w has shape [..., source, mic].
    w = tf.transpose(w, prefix + [start + 1, start])
    # w has shape [..., mic, source].
  else:
    raise ValueError('Unknown beamformer type {}.'.format(beamformer_type))
  return w


def _estimate_time_invariant_covariances(y: tf.Tensor,
                                         t: tf.Tensor,
                                         use_complex_mask: bool = False,
                                         refmic: int = 0,
                                         ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Find time-invariant covariance matrices from masks.

  The inputs are the mixture signal and source estimates.
  Args:
    y: Mixture signal with shape [batch, mic, frame, bin].
    t: Source estimates at reference mic [batch, source, frame, bin].
    use_complex_mask: If True, use a complex mask.
    refmic: Reference microphone index.
  Returns:
    y_ti_cov: time-invariant spatial covariance matrix for mixture signal of
      shape [batch, bin, mic, mic].
    t_ti_cov: time-invariant spatial covariance matrix for source signals of
      shape [batch, bin, mic, mic, source].
  """
  tensor_shaper = shaper.Shaper()
  tensor_shaper.register_axes(y, ['batch', 'mic', 'frame', 'bin'])
  y = tensor_shaper.change(y,
                           ['batch', 'mic', 'frame', 'bin'],
                           ['batch', 'frame', 'bin', 'mic', 1])
  t = tensor_shaper.change(t,
                           ['batch', 'source', 'frame', 'bin'],
                           ['batch', 'frame', 'bin', 'source'])
  y_outprod = tf.matmul(y, y, adjoint_b=True)
  tensor_shaper.register_axes(y_outprod,
                              ['batch', 'frame', 'bin', 'mic', 'mic'])
  y_ti_cov = tf.reduce_mean(y_outprod, axis=1)
  tensor_shaper.register_axes(y_ti_cov, ['batch', 'bin', 'mic', 'mic'])
  t_power = tf.square(tf.abs(t))

  if use_complex_mask:
    y_refmic = y[:, :, :, refmic:refmic+1, 0]
    y_refmic_power = tf.square(tf.abs(y_refmic))
    power_limit = 1e-8
    est_masks = tf.where(
        tf.logical_and(y_refmic_power > power_limit,
                       t_power < y_refmic_power * 3.0),
        t / (y_refmic + power_limit), tf.zeros_like(t))
    est_masks = tf.conj(est_masks)
  else:
    # Derive an estimated Wiener-like mask from estimated target powers.
    power_offset = 1e-8
    t_power += power_offset
    est_masks = t_power / tf.reduce_sum(t_power, axis=-1, keepdims=True)

  est_masks = tf.cast(est_masks, dtype=y_outprod.dtype)
  est_masks = tensor_shaper.change(est_masks,
                                   ['batch', 'frame', 'bin', 'source'],
                                   ['batch', 'frame', 'bin', 1, 1, 'source'])

  masked_y_outprod = tf.expand_dims(y_outprod, axis=-1) * est_masks
  tensor_shaper.register_axes(masked_y_outprod,
                              ['batch', 'frame', 'bin', 'mic', 'mic', 'source'])

  t_ti_cov = tf.reduce_mean(masked_y_outprod, axis=1)
  tensor_shaper.register_axes(t_ti_cov,
                              ['batch', 'bin', 'mic', 'mic', 'source'])

  return y_ti_cov, t_ti_cov


def time_invariant_multichannel_filtering(
    y: tf.Tensor,
    t: tf.Tensor,
    use_complex_mask: bool = False,
    beamformer_type: str = 'wiener',
    refmic: int = 0,
    diagload: float = 1e-3,
    epsilon: float = 1e-8,
    apply_postfilter: bool = False,
    )-> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a multi-channel Wiener filter from time-invariant covariances.

  Args:
    y: [batch, mic, frame, bin], complex64, mixture spectrogram.
    t: [batch, source, frame, bin], complex64, estimated spectrogram.
    use_complex_mask: If True, use a complex mask.
    beamformer_type: A string describing beamformer type. 'wiener', 'mvdr'
      or 'mpdr'.
    refmic: index of the reference mic.
    diagload: A float32 value, diagonal loading for the matrix inversion in
      beamforming.
    epsilon: A float32 value, data-independent stabilizer for diagonal loading.
    apply_postfilter: Apply a single-frame post filter to multiply beamformed
      signal to get closer to the target.

  Returns:
    bf_y: [batch, source, frame, bin], complex64, beamformed spectrogram.
    w_H: [batch, bin, source, mic], complex64, beamformer coefficients.
  """
  tensor_shaper = shaper.Shaper()
  tensor_shaper.register_axes(y, ['batch', 'mic', 'frame', 'bin'])
  with tf.name_scope(None, 'time_invariant_multichannel_wiener_filter'):
    # This method uses the spatial covariance of the mixture and target.
    # In this method we estimate the spatial covariances of the noisy and
    # target signals by applying the same mask to all channels of the noisy
    # signal.
    # In the future, when we support multi-channel targets, then this
    # approximation will no longer be needed, and we can directly compute the
    # target spatial covariances.

    y_ti_cov, t_ti_cov = _estimate_time_invariant_covariances(
        y, t, use_complex_mask, refmic)
    tensor_shaper.register_axes(y_ti_cov, ['batch', 'bin', 'mic', 'mic'])
    tensor_shaper.register_axes(t_ti_cov,
                                ['batch', 'bin', 'mic', 'mic', 'source'])
    w = _get_beamformer_from_covariances(
        y_ti_cov, t_ti_cov, diagload=diagload, epsilon=epsilon, refmic=refmic,
        beamformer_type=beamformer_type)
    w_h = tf.conj(tensor_shaper.change(w,
                                       ['batch', 'bin', 'mic', 'source'],
                                       ['batch', 'bin', 'source', 'mic']))
    y = tensor_shaper.change(y,
                             ['batch', 'mic', 'frame', 'bin'],
                             ['batch', 'bin', 'mic', 'frame'])
    w_h_y = tf.matmul(w_h, y)
    bf_y = tensor_shaper.change(w_h_y,
                                ['batch', 'bin', 'source', 'frame'],
                                ['batch', 'source', 'frame', 'bin'])
    if apply_postfilter:
      post_filter = filtering_projections.calculate_scalar_filter(bf_y, t)
      bf_y *= post_filter
      post_filter = tensor_shaper.change(post_filter,
                                         ['batch', 'source', 1, 'bin'],
                                         ['batch', 'bin', 'source', 1])
      w_h *= post_filter
  return bf_y, w_h


def _maybe_expand_context(y: tf.Tensor, frame_context_length: int,
                          frame_context_type: str,
                          refmic: int) -> tf.Tensor:
  """Expands context into mics dimension of the input.

  Args:
    y: [batch, mic, frame, bin]
    frame_context_length:
    frame_context_type:
    refmic: Reference microphone.
  Returns:
    expanded_y: [batch, mic * frame_context_length, frame, bin]
    refmic: Updated refmic.
  """
  if frame_context_length > 1:
    tensor_shaper = shaper.Shaper()
    tensor_shaper.register_axes(y, ['batch', 'mic', 'frame', 'bin'])
    # If frame_context_length > 1, we pad mic axis of y with its contextual
    # frame values.
    # New mics axis is going to be mics * frame_context_length size and
    # refmic will change.
    if frame_context_type == 'causal':
      y = tf.pad(y, [(0, 0), (0, 0), (frame_context_length-1, 0), (0, 0)])
      # Center frame is the last one in the causal context case.
      center_frame_index = frame_context_length - 1
    elif frame_context_type == 'centered':
      pad_end = (frame_context_length - 1) // 2
      pad_begin = frame_context_length - 1 - pad_end
      y = tf.pad(y, [(0, 0), (0, 0), (pad_begin, pad_end), (0, 0)])
      # Center frame has the index pad_begin in the centered context case.
      center_frame_index = pad_begin
    else:
      raise ValueError('Unknown frame context type '
                       '{}'.format(frame_context_type))
    y = tf.signal.frame(y, frame_context_length, 1, axis=2)
    y = tensor_shaper.change(y,
                             ['batch', 'mic', 'frame', 'context', 'bin'],
                             ['batch', ('mic', 'context'), 'frame', 'bin'])
    # New refmic index is given by:
    #   np.ravel_multi_index([refmic, center_frame_index],
    #                        [mics, frame_context_length])
    refmic = refmic * frame_context_length + center_frame_index

    return y, refmic
  else:
    return y, refmic


def compute_multichannel_filter(y: tf.Tensor, t: tf.Tensor,
                                use_complex_mask: bool = False,
                                frame_context_length: int = 1,
                                frame_context_type: str = 'causal',
                                beamformer_type: str = 'wiener',
                                refmic: int = 0,
                                block_size_in_frames: int = -1,
                                diagload: float = 1e-3,
                                epsilon: float = 1e-8,
                                apply_postfilter: bool = False,
                                ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a multi-channel Wiener filter from spectrogram-like inputs.

  Args:
    y: [batch, mic, frame, bin], complex64/float32, mixture spectrogram.
    t: [batch, source, frame, bin], complex64/float32, estimated spectrogram.
    use_complex_mask: If True, use a complex mask.
    frame_context_length: An integer value to specify the number of
      contextual frames used in beamforming.
    frame_context_type: 'causal' or 'centered'.
    beamformer_type: A string describing beamformer type. 'wiener', 'mvdr'
      or 'mpdr'.
    refmic: index of the reference mic.
    block_size_in_frames: an int32 value, block size in frames.
    diagload: float32, diagonal loading value for the matrix inversion in
      beamforming. Note that this value is likely dependent on the energy level
      of the input mixture. The default value has been tuned based on the
      assumption that the time-domain RMS normalization is performed, and the
      covariance matrices are always divided by the number of frames.
    epsilon: A float32 value, data-independent stabilizer for diagonal loading.
    apply_postfilter: Apply a single-frame post-filter after calculating
      beamforming coefficients.

  Returns:
    [batch, source, frame, bin], complex64/float32, beamformed y.
    [batch, bin, source, mic], complex64/float32, beamformer weights.
  """

  y = tf.convert_to_tensor(y, name='y')
  t = tf.convert_to_tensor(t, name='t')
  tensor_shaper = shaper.Shaper()
  tensor_shaper.register_axes(y, ['batch', 'mic', 'frame', 'bin'])
  tensor_shaper.register_axes(t, ['batch', 'source', 'frame', 'bin'])

  batch = tensor_shaper.axis_sizes['batch']
  n_frames = tensor_shaper.axis_sizes['frame']
  # Expand context into microphones axis and update reference microphone index.
  y, refmic = _maybe_expand_context(y, frame_context_length, frame_context_type,
                                    refmic)
  # y now has shape [batch, mic*context, frame, bin]

  if block_size_in_frames < 0:
    n_frames_in_block = n_frames
    perform_blocking = False
  else:
    assert block_size_in_frames > 0
    if tf.is_tensor(n_frames):
      n_frames_in_block = tf.minimum(n_frames, block_size_in_frames)
    else:
      n_frames_in_block = min(n_frames, block_size_in_frames)
    perform_blocking = True

  if perform_blocking:
    # This window is used before and after beamforming.
    overlap_window = tf.cast(tf.signal.vorbis_window(n_frames_in_block),
                             dtype=y.dtype)

    def extract_blocks(tensor):
      """Extract overlapping blocks from signals."""
      half_size = n_frames_in_block // 2
      tensor = tf.pad(tensor, [(0, 0), (0, 0), (half_size, 0), (0, 0)])
      tensor = tf.signal.frame(tensor, n_frames_in_block, half_size,
                               pad_end=True, axis=-2)
      local_shaper = shaper.Shaper()
      tensor = local_shaper.change(
          tensor,
          ['batch', 'chan', 'block', 'frame', 'bin'],
          [('batch', 'block'), 'chan', 'frame', 'bin'])
      window_reshaped = tf.reshape(overlap_window, [1, 1, n_frames_in_block, 1])
      tensor *= window_reshaped
      return tensor

    y = extract_blocks(y)
    # y has shape [n_blocks*batch, mic, n_frames_in_block, bin].
    t = extract_blocks(t)
    # t has shape [n_blocks*batch, source, n_frames_in_block, bin].

  bf_y, beamformer_weights = time_invariant_multichannel_filtering(
      y, t, use_complex_mask=use_complex_mask,
      beamformer_type=beamformer_type, refmic=refmic, diagload=diagload,
      epsilon=epsilon, apply_postfilter=apply_postfilter)
  # bf_y has shape [n_blocks*batch, source, n_frames_in_block, bin].
  # or bf_y has shape [batch, source, frame, bin].
  # beamformer_weights has shape [batch, bin, source, mic]

  if perform_blocking:
    block_shaper = shaper.Shaper()
    block_shaper.register_axes(bf_y,
                               ['block_and_batch', 'source',
                                'frame_in_block', 'bin'])
    half_size = n_frames_in_block // 2
    n_blocks = tf.shape(bf_y)[0] / batch
    # Overlap add overlapping blocks.
    tensor_shape = tf.concat([[n_blocks, batch], tf.shape(bf_y)[1:]], axis=0)
    bf_y = tf.reshape(bf_y, tensor_shape)
    block_shaper.register_axes(
        bf_y, ['block', 'batch', 'source', 'frame_in_block', 'bin'])
    # bf_y has shape [n_blocks, batch, source, block_size, bin].
    bf_y = block_shaper.change(
        bf_y, ['block', 'batch', 'source', 'frame_in_block', 'bin'],
        ['batch', 'source', 'bin', 'block', 'frame_in_block'])
    window_reshaped = tf.reshape(overlap_window,
                                 [1, 1, 1, 1, n_frames_in_block])
    # Window the beamformed signal, so that we end up applying window squared.
    bf_y *= window_reshaped
    # Overlap-add windowed overlapping blocks.
    bf_y = tf.signal.overlap_and_add(bf_y, half_size)
    # bf_y has shape [batch, source, bin, frame_padded]
    bf_y = bf_y[..., half_size:half_size + n_frames]
    block_shaper.register_axes(
        bf_y, ['batch', 'source', 'bin', 'frame'])
    bf_y = block_shaper.change(bf_y,
                               ['batch', 'source', 'bin', 'frame'],
                               ['batch', 'source', 'frame', 'bin'])

  return bf_y, beamformer_weights


def compute_multichannel_filter_from_signals(y: tf.Tensor, t: tf.Tensor,
                                             refmic: int = 0,
                                             sample_rate: float = 16000.,
                                             ws: float = 0.064,
                                             hs: float = 0.032,
                                             frame_context_length: int = 1,
                                             frame_context_type: str = 'causal',
                                             beamformer_type: str = 'wiener',
                                             block_size_in_seconds: int = -1,
                                             use_complex_mask: bool = False,
                                             diagload: float = 1e-3,
                                             epsilon: float = 1e-8,
                                             n_fft: int = -1,
                                             window_fn_name: str = 'sqrt_hann',
                                             apply_postfilter: bool = False,
                                             ) -> tf.Tensor:
  """Computes a multichannel Wiener filter to estimate a target t from y.

  Args:
    y: [batch, mic, time], float32, mixture waveform.
    t: [batch, source, time], float32, estimated waveform.
    refmic: Index of the reference mic.
    sample_rate: Sampling rate of audio in Hz.
    ws: Window size in seconds.
    hs: Hop size in seconds.
    frame_context_length: An integer value to specify the number of
      contextual frames used in beamforming.
    frame_context_type: 'causal' or 'centered'.
    beamformer_type: A string describing beamformer type. 'wiener', 'mvdr'
      or 'mpdr'.
    block_size_in_seconds: block size in seconds.
    use_complex_mask: If True, use a complex mask.
    diagload: float32, diagonal loading value for the matrix inversion in
      beamforming. Note that this value is likely dependent on the energy level
      of the input mixture. The default value has been tuned based on the
      assumption that the time-domain RMS normalization is performed, and the
      covariance matrices are always divided by the number of frames.
    epsilon: A float32 value, data-independent stabilizer for diagonal loading.
    n_fft: If > 0, specifies the FFT size to use for STFT
      calculation. Otherwise, fft size is calculated automatically.
    window_fn_name: STFT analysis window function name.
    apply_postfilter: If True, apply a single-frame post-filter after
      beamforming.

  Returns:
    [batch, source, time], float32, beamformed waveform y.
  """

  noisy_length = signal_util.static_or_dynamic_dim_size(y, -1)

  # Compute transforms.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=sample_rate,
      window_time_seconds=ws,
      hop_time_seconds=hs,
      magnitude_offset=1e-8,
      zeropad_beginning=True,
      num_basis=n_fft,
      window_fn_name=window_fn_name,
  )
  y_spectrograms = transformer.forward(y)
  t_spectrograms = transformer.forward(t)

  block_size_in_frames = int(round(block_size_in_seconds / hs))

  # Perform beamforming.
  beamformed_spectrograms, _ = compute_multichannel_filter(
      y_spectrograms, t_spectrograms,
      frame_context_length=frame_context_length,
      frame_context_type=frame_context_type,
      beamformer_type=beamformer_type,
      refmic=refmic,
      block_size_in_frames=block_size_in_frames,
      use_complex_mask=use_complex_mask,
      diagload=diagload,
      epsilon=epsilon,
      apply_postfilter=apply_postfilter)

  # Reconstruct time-domain signals.
  beamformed_waveforms = transformer.inverse(
      beamformed_spectrograms)[..., :noisy_length]

  return beamformed_waveforms


@dataclasses.dataclass(frozen=True)
class BeamformerParams(object):
  """Dataclass for estimating and applying an LTI beamformer.

  The parameters are related to how to transform a time-domain signal to STFT
  domain and in what manner to estimate an LTI beamformer from that. A subset
  of the parameters (namely the first 7 hyper-parameters) are required when
  applying the estimated beamformer to another time-domain signal since we
  need to know how to transform the time domain signal to the STFT domain and
  how much frame context we would like to use.

  Attributes:
    sample_rate: Sample rate.
    ws: Window size for STFT in seconds.
    hs: Hop size for STFT in seconds.
    n_fft: If > 0, specifies the fft size for STFT calculation. Otherwise,
      it is calculated by SignalTransformer automatically as the enclosing
      power of 2 of the window size (or frame length) in samples.
    window_fn_name: The name of the analysis window function to use in STFT.
    frame_context_length: Length of context in frames.
    frame_context_type: 'centered' or 'causal'.
    refmic: Reference microphone.
    beamformer_type: 'wiener', 'mvdr' or 'mpdr'.
    use_complex_mask: If True, use a complex mask. If False, we use a real mask
      which is a Wiener-like mask. If False, the target sources should
      include all sources in the mixture, not only some of them, otherwise the
      beamformer will be incorrect.
    apply_postfilter: If True, apply a single-frame post-filter to beamformed
      signals.
    diagload: float32, diagonal loading value for the matrix inversion in
      beamforming. Note that this value is likely dependent on the energy level
      of the input mixture. The default value has been tuned based on the
      assumption that the time-domain RMS normalization is performed, and the
      covariance matrices are always divided by the number of frames.
    epsilon: A float32 value, data-independent stabilizer for diagonal loading.
  """
  sample_rate: float = 16000.
  ws: float = 0.064
  hs: float = 0.032
  n_fft: int = -1
  window_fn_name: str = 'sqrt_hann'
  frame_context_length: int = 1
  frame_context_type: str = 'causal'
  refmic: int = 0
  beamformer_type: str = 'wiener'
  use_complex_mask: bool = True
  apply_postfilter: bool = False
  diagload: float = 1e-3
  epsilon: float = 1e-8


def compute_lti_beamformer_from_signals(
    y: tf.Tensor,
    t: tf.Tensor,
    beamformer_params: BeamformerParams = BeamformerParams(),
    ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a beamformer to estimate target(s) t from multi-channel y.

  Note that when beamformer_params.use_complex_mask=False, we assume that the
  source dimension in the target t tensor contains sources that approximately
  sum to the mixture waveform at the reference microphone, so that we can
  calculate a Wiener-like mask using those target sources. If this is not true,
  that is if t does not include all sources but only one of them or some of
  them, we should set beamformer_parems.use_complex_mask=True.

  Args:
    y: [batch, mic, time], float32, mixture waveform.
    t: [batch, source, time], float32, estimated waveform.
    beamformer_params: Beamformer parameters needed to apply the beamformer
      to time domain signals.

  Returns:
    [batch, bin, source, mic]: multichannel multiframe beamformer coefficients.
      Note that the mic axis also includes contextual frames if
      frame_context_length > 1 and real-imaginary parts if concat_real_imag is
      True. If concat_real_imag is True, source dimension is 2*source including
      real and imaginary for each source.
    [batch, source, time], float32, beamformed waveform y.
  """

  noisy_length = signal_util.static_or_dynamic_dim_size(y, -1)

  # Compute transforms.
  transformer = signal_transformer.SignalTransformer(
      sample_rate=beamformer_params.sample_rate,
      window_time_seconds=beamformer_params.ws,
      hop_time_seconds=beamformer_params.hs,
      magnitude_offset=1e-8,
      zeropad_beginning=True,
      num_basis=beamformer_params.n_fft,
      window_fn_name=beamformer_params.window_fn_name,
  )
  y_spectrograms = transformer.forward(y)
  t_spectrograms = transformer.forward(t)

  # Perform beamforming.
  beamformed_spectrograms, beamformer_weights = compute_multichannel_filter(
      y_spectrograms, t_spectrograms,
      frame_context_length=beamformer_params.frame_context_length,
      frame_context_type=beamformer_params.frame_context_type,
      beamformer_type=beamformer_params.beamformer_type,
      refmic=beamformer_params.refmic,
      block_size_in_frames=-1,
      use_complex_mask=beamformer_params.use_complex_mask,
      diagload=beamformer_params.diagload,
      epsilon=beamformer_params.epsilon,
      apply_postfilter=beamformer_params.apply_postfilter)

  # Reconstruct time-domain signals through inverse STFT.
  beamformed_waveforms = transformer.inverse(
      beamformed_spectrograms)[..., :noisy_length]

  return beamformer_weights, beamformed_waveforms


def apply_lti_beamformer_to_signal(
    y: tf.Tensor,
    beamformer_weights: tf.Tensor,
    beamformer_params: BeamformerParams = BeamformerParams(),
    ) -> tf.Tensor:

  """Applies a beamformer to signal(s) in y.

  Args:
    y: [batch, mic, time], float32, multi-channel input waveform.
    beamformer_weights: [batch, bin, source, mic_and_context],
      float32/complex64, beamformer coefficients. Batch dimensions should
      broadcast with the tensor y.
    beamformer_params: Beamformer hyper parameters required to apply to a
      new signal.

  Returns:
    [batch, source, time], float32, beamformed waveforms.
  """
  input_length = signal_util.static_or_dynamic_dim_size(y, -1)
  transformer = signal_transformer.SignalTransformer(
      sample_rate=beamformer_params.sample_rate,
      window_time_seconds=beamformer_params.ws,
      hop_time_seconds=beamformer_params.hs,
      magnitude_offset=1e-8,
      zeropad_beginning=True,
      num_basis=beamformer_params.n_fft,
      window_fn_name=beamformer_params.window_fn_name,
  )
  y_stft = transformer.forward(y)

  y_stft, _ = _maybe_expand_context(y_stft,
                                    beamformer_params.frame_context_length,
                                    beamformer_params.frame_context_type,
                                    beamformer_params.refmic)
  tensor_shaper = shaper.Shaper()
  tensor_shaper.register_axes(y_stft, ['batch', 'mic_context', 'frame', 'bin'])
  y_stft = tensor_shaper.change(y_stft,
                                ['batch', 'mic_context', 'frame', 'bin'],
                                ['batch', 'bin', 'mic_context', 'frame'])
  beamformed_stft = tf.matmul(beamformer_weights, y_stft)
  beamformed_stft = tensor_shaper.change(beamformed_stft,
                                         ['batch', 'bin', 'source', 'frame'],
                                         ['batch', 'source', 'frame', 'bin'])

  # Reconstruct time-domain signals through inverse STFT.
  beamformed_waveforms = transformer.inverse(
      beamformed_stft)[..., :input_length]

  return beamformed_waveforms


def find_equivalent_time_domain_beamformer(
    beamformer_stft: tf.Tensor,
    beamformer_params: BeamformerParams = BeamformerParams(),
    filter_len: Optional[int] = None,
    filter_support_start: Optional[int] = None,
    random_seed: int = 0,
    ) -> Tuple[tf.Tensor, int]:
  """Finds an equivalent time domain beamformer to an STFT domain one.

  To get the exact same filter, use the same random_seed each time.

  Args:
    beamformer_stft: An STFT domain beamformer with shape
      [batch, bin, source, mic_and_context] and dtype tf.complex64.
    beamformer_params: Beamformer parameters.
    filter_len: Filter length in samples. If None, it is calculated
      automatically.
    filter_support_start: Filter support start index. If None, it is calculated
      automatically.
    random_seed: Random seed to use for generating WGN input.
  Returns:
    beamformer_time_domain: A beamformer [batch, source, mic, filter_len] of
      dtype tf.float32.
    filter_support_start: The starting sample index of the time domain
      filter. A value of 0 implies a causal filter, a value of -5 means the
      filter is non-causal where its values start from index -5.
  """
  # bf weights dimensions are [batch, bin, source, mic_and_context]
  bf_shape = _smart_shape(beamformer_stft)
  batch_size = bf_shape[0]
  num_sources = bf_shape[2]
  mics = bf_shape[-1] // beamformer_params.frame_context_length
  new_shape = bf_shape[:-1] + [mics, beamformer_params.frame_context_length]
  # new_shape is [batch, bin, source, mic, context]
  beamformer_weights_expanded = tf.reshape(beamformer_stft, new_shape)
  frame_shift_samples = int(round(
      beamformer_params.hs * beamformer_params.sample_rate))
  frame_length_samples = int(round(
      beamformer_params.ws * beamformer_params.sample_rate))
  # Time-domain filter should have a length that increases with frame context
  # length with a minimum length of two frame lengths long.
  if filter_len is None:
    filter_len = ((beamformer_params.frame_context_length - 1) *
                  frame_shift_samples + 2 * frame_length_samples)
  # An STFT domain filter will have a lookahead which is equal to the
  # frame length.
  if filter_support_start is None:
    if beamformer_params.frame_context_type == 'causal':
      filter_support_start = -frame_length_samples
    elif beamformer_params.frame_context_type == 'centered':
      filter_support_start = (
          -frame_length_samples - (
              ((beamformer_params.frame_context_length - 1) // 2) *
              frame_shift_samples))

  time_domain_filters = []
  tf.set_random_seed(random_seed)
  for i in range(mics):
    # We use a single mic beamformer and have the context axis behave as
    # mics_and_context axis for beamforming a single mic input.
    bf_weights_for_mic_i = beamformer_weights_expanded[:, :, :, i, :]
    # The strategy is to filter a random signal with STFT domain filter and then
    # find an equivalent LTI time-domain filter that would yield a close result.
    # Note: We can use a more carefully designed input signal here, but using a
    # WGN random signal works OK.
    input_signal_length = filter_len * frame_shift_samples
    # We have a single mic input signal which is WGN.
    x = tf.random.normal([batch_size, 1, input_signal_length])
    y = apply_lti_beamformer_to_signal(x, bf_weights_for_mic_i,
                                       beamformer_params=beamformer_params)
    # The output y has shape [batch, source, time].
    for source in range(num_sources):
      _, time_domain_filter = filtering_projections.filter_in_time_domain(
          x, y[:, source:source+1, :],
          filter_support_start=filter_support_start, filter_len=filter_len,
          solver='default')
      time_domain_filters.append(time_domain_filter)

  beamformer_time_domain = tf.concat(time_domain_filters, axis=-2)
  # beamformer_time_domain has shape [batch, mics*sources, filter_len].
  beamformer_time_domain = tf.reshape(
      beamformer_time_domain, [batch_size, mics, num_sources, filter_len])
  beamformer_time_domain = tf.transpose(
      beamformer_time_domain, (0, 2, 1, 3))  # [batch, source, mic, filter_len]
  return beamformer_time_domain, filter_support_start


def apply_time_domain_beamformer(input_signal: tf.Tensor,
                                 beamformer_time_domain: tf.Tensor,
                                 filter_support_start: int = 0
                                 ) -> tf.Tensor:
  """Applies a time domain beamformer to signal.

  Args:
    input_signal: Input signal with shape (batch, mic, signal_length).
    beamformer_time_domain: Beamformer with shape
      (batch, source, mic, filter_length).
    filter_support_start: The starting sample index of the time domain
      filter. A value of 0 implies a causal filter, a value of -5 means the
      filter is non-causal where its values start from index -5.
  Returns:
    output_signal: Output beamformed signal with shape
      (batch, source, signal_length).
  """
  num_sources = _smart_shape(beamformer_time_domain)[1]
  bf_outputs = []
  for source in range(num_sources):
    filtered_output = filtering_projections.perform_filtering(
        input_signal, beamformer_time_domain[:, source],
        filter_support_start=filter_support_start)
    bf_output = tf.reduce_sum(filtered_output, axis=-2, keepdims=True)
    bf_outputs.append(bf_output)
  return tf.concat(bf_outputs, axis=-2)
