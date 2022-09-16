"""Filtering projections.

Finds the best filter that minimizes the difference between filtered input
signal and a provided target signal using various domains of filtering.

Filtering can be done in time domain, single-frame STFT domain,
multi-frame STFT domain or single-frame double-STFT domain.
"""

import functools
from typing import Tuple, Callable, Optional, Any, Sequence, Union
import numpy as np
import tensorflow as tf


def _log2(values: tf.Tensor) -> tf.Tensor:
  return tf.math.log(values) / tf.math.log(2.0)


def _sqrt_hann(*args, **kwargs) -> tf.Tensor:
  return tf.sqrt(tf.signal.hann_window(*args, **kwargs))


def _enclosing_power_of_2(input_len: Union[int, tf.Tensor]) -> tf.Tensor:
  """Returns the smallest power of 2 greater than or equal to input_len."""
  n = tf.math.ceil(_log2(tf.cast(input_len, tf.float32) - 0.5))
  return tf.cast(tf.round(tf.pow(2., tf.cast(n, tf.float32))), tf.int32)


def _complex_to_realimag(matrix: tf.Tensor) -> tf.Tensor:
  """Converts a complex MxN matrix to a 2Mx2N real matrix."""
  matrix_real = tf.math.real(matrix)
  matrix_imag = tf.math.imag(matrix)
  return tf.concat([
      tf.concat([matrix_real, -matrix_imag], axis=-1),
      tf.concat([matrix_imag, matrix_real], axis=-1)
  ], axis=-2)


def _smart_dim(tensor: tf.Tensor, i: int):
  """Static or dynamic size for dimension `i`."""
  static_shape = tensor.shape
  dynamic_shape = tf.shape(tensor)
  return (static_shape[i].value if hasattr(static_shape[i], 'value')
          else static_shape[i]) or dynamic_shape[i]


def _least_squares_cholesky(matrix, rhs):
  cov_matrix = tf.matmul(matrix, matrix, adjoint_a=True)
  cov_matrix += 1e-5 * tf.eye(_smart_dim(cov_matrix, -1),
                              dtype=cov_matrix.dtype)
  return tf.linalg.cholesky_solve(tf.linalg.cholesky(cov_matrix),
                                  tf.matmul(matrix, rhs, adjoint_a=True))


def _real_solve(matrix: tf.Tensor, rhs: tf.Tensor,
                real_solver: Callable[..., Any]) -> tf.Tensor:
  """Solves real/complex matrix inversion problem with a real solver."""
  if matrix.dtype.is_floating and rhs.dtype.is_floating:
    return real_solver(matrix, rhs)
  matrix_realimag = _complex_to_realimag(matrix)
  n = _smart_dim(matrix, -1)
  rhs_realimag = tf.concat([tf.math.real(rhs), tf.math.imag(rhs)], axis=-2)
  lhs_realimag = real_solver(matrix_realimag, rhs_realimag)
  return tf.dtypes.complex(lhs_realimag[..., :n, :], lhs_realimag[..., n:, :])


def _matrix_solve(matrix: tf.Tensor, rhs: tf.Tensor, method: str = 'default',
                  ) -> tf.Tensor:
  """Matrix_solve using various methods.

  Args:
    matrix: (..., M, N).
    rhs: (..., M, K).
    method: 'default', 'ls_cholesky', 'cholesky' or 'lstsq'.
  Returns:
    lhs: (..., N, K).
  """
  solve_fns = {
      'default': tf.linalg.solve,  # For square matrices, M=N.
      # For rectangular or square matrices.
      'lstsq': functools.partial(tf.linalg.lstsq, fast=False),
      'ls_cholesky': _least_squares_cholesky,  # For LS on TPU.
      # Cholesky only for Hermitian matrices. pylint: disable=g-long-lambda
      'cholesky': lambda m, r: tf.linalg.cholesky_solve(
          tf.linalg.cholesky(m), r),
  }
  if method in solve_fns:
    return _real_solve(matrix, rhs, solve_fns[method])
  else:
    raise ValueError(f'Unknown matrix solve method {method}.')


def _pad_beginning(waveform: tf.Tensor, pad_len: int) -> tf.Tensor:
  """Pad with zeros at the beginning of last axis with pad_len amount."""
  pad_spec = [(0, 0)] * (len(waveform.shape) - 1) + [(pad_len, 0)]
  return tf.pad(waveform, pad_spec)


def _clip_beginning(waveform: tf.Tensor, clip_len: int) -> tf.Tensor:
  """Remove clip_len elements from the beginning at the last axis."""
  return waveform[..., clip_len:]


def _stft(signal: tf.Tensor, frame_length: int, frame_step: int,
          nfft: Optional[int] = None,
          window_fn: Callable[..., tf.Tensor] = _sqrt_hann,
          ) -> tf.Tensor:
  """Performs STFT with padding in the beginning and at the end."""
  if nfft is not None and nfft < frame_length:
    raise ValueError(f'nfft={nfft} is smaller than frame_length={frame_length}')
  return tf.signal.stft(
      _pad_beginning(signal, frame_length - frame_step), frame_length,
      frame_step, fft_length=nfft, pad_end=True, window_fn=window_fn)


def _istft(signal_stft: tf.Tensor, frame_length: int, frame_step: int,
           nfft: Optional[int] = None,
           window_fn: Callable[..., tf.Tensor] = _sqrt_hann,
           output_len: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Performs inverse STFT with padding in the beginning and at the end."""
  if nfft is not None and nfft < frame_length:
    raise ValueError(f'nfft={nfft} is smaller than frame_length={frame_length}')
  signal = tf.signal.inverse_stft(
      signal_stft, frame_length, frame_step, fft_length=nfft,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step, window_fn))
  signal = _clip_beginning(signal, frame_length - frame_step)
  if output_len is not None:
    signal = signal[..., :output_len]
  return signal


# Next two functions are copied/modified from jeremyt@'s CL 354291790.
def _pad_to_length(a: tf.Tensor, length: Union[int, tf.Tensor]) -> tf.Tensor:
  """Zero pad last dimension of a at the end to achieve length."""
  padding = length - _smart_dim(a, -1)
  padding = tf.expand_dims(tf.expand_dims(padding, 0), 0)  # shape [1, 1]
  padding = tf.pad(padding, [[tf.rank(a) - 1, 0], [1, 0]])  # shape [rank, 2]
  return tf.pad(a, padding)


def _get_delta_filter(filt_shape, filter_support_start, filt_dtype):
  """Returns a delta filter that acts like an identity function."""
  filt_pre_shape = filt_shape[:-1]
  filter_len = filt_shape[-1]
  one_hot_loc = tf.maximum(-filter_support_start, 0)
  one_hot_loc_tensor = one_hot_loc * tf.ones(filt_pre_shape, dtype=tf.int32)
  return tf.cast(tf.one_hot(one_hot_loc_tensor, filter_len), filt_dtype)


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


def calculate_scalar_filter(input_stft: tf.Tensor, target_stft: tf.Tensor,
                            frame_skip: int = 1) -> tf.Tensor:
  """Calculates scalar complex filter from input_stft to target_stft.

  The scalar filter makes the filtered input closest to the target.

  Args:
    input_stft: Input STFT domain tensor with frames axis on axis=-2. All other
      axes are treated as batch axes.
    target_stft: Target STFT domain tensor with frames axis on axis=-2.
    frame_skip: Skip this many frames from beginning and the end when
      calculating autocorrelation and cross-correlations.
  Returns:
    scalar (complex) filter, with unit dimension on axis=-2 broadcastable to
      the shape of the input_stft.
  """
  # Note: We assume frames axis is axis=-2.
  input_stft = input_stft[..., frame_skip:-frame_skip, :]
  target_stft = target_stft[..., frame_skip:-frame_skip, :]
  r_autocorr = tf.reduce_sum(tf.math.conj(input_stft) * input_stft, axis=-2,
                             keepdims=True)
  r_crosscorr = tf.reduce_sum(tf.math.conj(input_stft) * target_stft, axis=-2,
                              keepdims=True)
  return r_crosscorr / (r_autocorr + 1e-8)


def perform_filtering(input_signal: tf.Tensor, filt: tf.Tensor,
                      filter_support_start: int = 0) -> tf.Tensor:
  """Filters input_signal with filt when filt starts at a given index.

  Args:
    input_signal: Input signal with shape (..., signal_len).
    filt: Filter with shape (..., filter_len).
    filter_support_start: Starting index of the filter. The output obtained
      through inverse fft is circularly shifted to get the correct
      time-domain result which is of signal_len length.
      When filter_support_start=0, no shifting is required.
  Returns:
    filtered_signal: with shape (..., signal_len).
  """
  signal_len = _smart_dim(input_signal, -1)
  filter_len = _smart_dim(filt, -1)
  out_len = signal_len + filter_len - 1
  out_range = -filter_support_start + tf.range(signal_len)
  out_range = tf.where(out_range < 0, out_len + out_range, out_range)
  filtered_signal = convolve(input_signal, filt)
  return tf.gather(filtered_signal, out_range, axis=-1)


def _scatter_update_last_axis(tensor: tf.Tensor,
                              indices: Union[Sequence[int], tf.Tensor],
                              updates: tf.Tensor,
                              ) -> tf.Tensor:
  """Performs scatter_nd_update with indices and updates for the last axis.

  Args:
    tensor: Input tensor of dimension (..., full_size).
    indices: tensor of shape (sub_size,) with indices into the last
      dimension of the input tensor.
    updates: tensor of shape either (..., sub_size) where ... matches the
      input tensor shape or (sub_size,) with values to update the last axis
      of the tensor at the provided indices.
  Returns:
    Updated tensor with the same shape as input (..., full_size).
  """
  num_updates = _smart_dim(updates, -1)
  tensor_shape = tf.shape(tensor)
  # First, we reshape tensor to have a shape of [batch, last_dim].
  tensor_reshaped = tf.reshape(tensor, [-1, tensor_shape[-1]])
  batch_size = _smart_dim(tensor_reshaped, 0)
  batch_tiled = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
                        [1, num_updates])
  indices_tiled = tf.tile(tf.expand_dims(indices, axis=0), [batch_size, 1])
  updates_rank = len(updates.shape)
  if updates_rank > 1:
    assert updates_rank == len(tensor.shape)
    updates_tiled = tf.reshape(updates, (batch_size, num_updates))
  elif updates_rank == 1:
    updates_tiled = tf.tile(tf.expand_dims(updates, axis=0), [batch_size, 1])
  batch_tiled_flat = tf.reshape(batch_tiled, [-1])
  indices_tiled_flat = tf.reshape(indices_tiled, [-1])
  updates_tiled_flat = tf.reshape(updates_tiled, [-1])
  indices = tf.stack((batch_tiled_flat, indices_tiled_flat), axis=1)
  tensor_reshaped = tf.tensor_scatter_nd_update(tensor_reshaped, indices,
                                                updates_tiled_flat)
  return tf.reshape(tensor_reshaped, tensor_shape)


def _solve_with_fixed_indices(matrix: tf.Tensor, rhs: tf.Tensor,
                              fixed_indices: Sequence[int],
                              fixed_values: Sequence[float],
                              solver: str = 'ls_cholesky',
                              ) -> tf.Tensor:
  """Least-squares minimize ||M x - rhs|| for x, when some part of x is fixed.

  Args:
    matrix: Input matrix M in M x.
    rhs: Right hand side in M x - rhs.
    fixed_indices: Fixed indices of vector x.
    fixed_values: Fixed values corresponding to fixed_indices of x.
    solver: The string describing the solver. Since the solver needs to work
      for non-square matrices, we need to use 'ls_cholesky' or 'lstsq'.
  Returns:
    Solution x.
  """
  assert solver == 'lstsq' or solver == 'ls_cholesky'
  fixed_len = len(fixed_indices)
  assert fixed_len == len(fixed_values)
  matrix_shape = tf.shape(matrix)
  solution_len = matrix_shape[-1]
  unfixed_indices = [i for i in range(solution_len) if i not in
                     list(fixed_indices)]
  batch_dims = len(matrix.shape) - 1
  solution_shape = tf.concat((matrix_shape[:-2], [solution_len]), axis=0)
  fixed_part_shape = tf.concat((matrix_shape[:-1], [fixed_len]), axis=0)
  fixed_value_part_shape = tf.concat((matrix_shape[:-2], [fixed_len, 1]),
                                     axis=0)
  unfixed_part_shape = tf.concat((matrix_shape[:-1],
                                  [tf.shape(unfixed_indices)[0]]), axis=0)
  fixed_indices_b = tf.broadcast_to(fixed_indices, fixed_part_shape)
  fixed_values = tf.cast(fixed_values, matrix.dtype)
  fixed_values_b = tf.broadcast_to(tf.reshape(fixed_values, [-1, 1]),
                                   fixed_value_part_shape)
  unfixed_indices_b = tf.broadcast_to(unfixed_indices, unfixed_part_shape)
  matrix_fixed_part = tf.gather(matrix, fixed_indices_b,
                                axis=-1, batch_dims=batch_dims)
  matrix_unfixed_part = tf.gather(matrix, unfixed_indices_b,
                                  axis=-1, batch_dims=batch_dims)
  rhs -= tf.matmul(matrix_fixed_part, fixed_values_b)[..., 0]
  solution_unfixed = _matrix_solve(
      matrix_unfixed_part, tf.expand_dims(rhs, -1), solver)[..., 0]
  solution = tf.zeros(solution_shape, matrix.dtype)
  solution = _scatter_update_last_axis(solution, unfixed_indices,
                                       solution_unfixed)
  solution = _scatter_update_last_axis(solution, fixed_indices, fixed_values)
  return solution


def calculate_multitap_filter(input_signal: tf.Tensor, target_signal: tf.Tensor,
                              frame_skip: int = 0,
                              filter_support_start: int = 0,
                              filter_len: int = 10,
                              diagload: float = 0.1,
                              fixed_indices: Optional[Sequence[int]] = None,
                              fixed_values: Optional[Sequence[float]] = None,
                              use_dense_solver: bool = True,
                              solver: str = 'default',
                              fix_nans: bool = False) -> tf.Tensor:
  """Calculates the multitap filter that takes input_signal closest to target.

  A filter is found and returned such that ||filt * input - target|| is
  minimized.  Here * is the convolution operation and the norm is the L2 norm.

  Args:
    input_signal: (..., signal_len) the signal to filter.
    target_signal: (..., signal_len) the signal to get close to.
    frame_skip: Skip this many frames from the start and the end.
    filter_support_start: The time index where the FIR filter support starts.
      For causal filters, this is 0. If filter_support_start < 0, then it
      is not a causal FIR filter, if filter_support_start > 0, then it causes
      a delay in input_signal after filtering.
    filter_len: FIR filter length.
    diagload: Diagonal loading to stabilize matrix inversion.
    fixed_indices: Fixed filter indices within the filter array, regardless of
      filter_support_start, values must be between 0 and filter_len-1.
    fixed_values: Fixed values for the given fixed indices.
    use_dense_solver: If True, use dense matrix solver, otherwise use Toeplitz
      solver.
    solver: The type of matrix solver when using a dense solver: 'lstsq',
      'default', 'cholesky', 'ls_cholesky'.
    fix_nans: If True and the filter has NaNs, the filter is replaced with an
      identity (or delta function) filter.
  Returns:
    filt: The FIR filter (..., filter_len) that takes input to target.
  """
  target_signal = tf.convert_to_tensor(target_signal)
  input_signal = tf.convert_to_tensor(input_signal)
  if target_signal.dtype.is_floating and input_signal.dtype.is_floating:
    forward_fft = tf.signal.rfft
    inverse_fft = tf.signal.irfft
    input_signal = tf.cast(input_signal, dtype=tf.float32)
    target_signal = tf.cast(target_signal, dtype=tf.float32)
  elif target_signal.dtype.is_complex or input_signal.dtype.is_complex:
    forward_fft = tf.signal.fft
    inverse_fft = tf.signal.ifft
    input_signal = tf.cast(input_signal, dtype=tf.complex64)
    target_signal = tf.cast(target_signal, dtype=tf.complex64)
  else:
    raise ValueError('Cannot determine filter type: real or complex.')

  if frame_skip > 0:
    input_signal = input_signal[..., frame_skip:-frame_skip]
    target_signal = target_signal[..., frame_skip:-frame_skip]
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
  rhs_range = tf.range(filter_support_start, filter_support_start + filter_len)
  rhs_range = tf.where(rhs_range < 0, fft_len + rhs_range, rhs_range)
  rhs = tf.gather(cross_corr, rhs_range, axis=-1)

  if use_dense_solver or fixed_indices is not None:
    dense_toeplitz_matrix = toeplitz_matrix.to_dense()

    if fixed_indices is not None:
      filt = _solve_with_fixed_indices(dense_toeplitz_matrix, rhs,
                                       fixed_indices, fixed_values)
    else:
      filt = _matrix_solve(dense_toeplitz_matrix,
                           tf.expand_dims(rhs, -1), solver)[..., 0]
  else:
    # This does not work on TPU, so set use_dense_solver=True on TPU.
    return toeplitz_matrix.solvevec(rhs)
  if fix_nans:
    delta_filter = _get_delta_filter(tf.shape(filt), filter_support_start,
                                     filt.dtype)
    filt_norm = tf.math.sqrt(tf.reduce_sum(tf.math.abs(filt)**2,
                                           axis=-1, keepdims=True))
    problematic = tf.math.is_nan(filt_norm)
    filt = tf.where(tf.broadcast_to(problematic, tf.shape(filt)),
                    delta_filter, filt)
  return filt


def nap_project_multitap_filter(filt: tf.Tensor, alpha: float = 0.7,
                                iterations: int = 1,
                                allow_scaling: bool = False) -> tf.Tensor:
  """Near-all-pass projection for a multitap filter."""
  if filt.dtype.is_floating:
    forward_fft = tf.signal.rfft
    inverse_fft = tf.signal.irfft
  elif filt.dtype.is_complex:
    forward_fft = tf.signal.fft
    inverse_fft = tf.signal.ifft
  else:
    raise ValueError(f'Unsupported filter type {filt.dtype}')
  filter_len = _smart_dim(filt, -1)
  filter_fft_len = _enclosing_power_of_2(filter_len)
  for _ in range(iterations):
    filt_fft = forward_fft(_pad_to_length(filt, filter_fft_len))
    filt_fft_mag = tf.math.abs(filt_fft)
    if allow_scaling:
      mean_mag = tf.reduce_mean(filt_fft_mag)
    else:
      mean_mag = 1.0
    norm_alpha_min = mean_mag * alpha
    norm_alpha_max = mean_mag * 1.0/alpha
    filt_fft_mag_clipped = tf.clip_by_value(filt_fft_mag, norm_alpha_min,
                                            norm_alpha_max)
    filt_projected = (tf.cast(filt_fft_mag_clipped, tf.complex64) *
                      tf.math.exp(tf.dtypes.complex(0.0, 1.0) *
                                  tf.cast(tf.math.angle(filt_fft),
                                          tf.complex64)))
    filt = inverse_fft(filt_projected)[..., :filter_len]
  return filt


def nap_project_scalar_filter(filt: tf.Tensor,
                              near_all_pass_alpha: float = 0.7) -> tf.Tensor:
  """Limit magnitude of the complex scalar between nap_alpha and 1/nap_alpha."""
  filt_mag = tf.math.abs(filt)
  filt_phase = tf.math.angle(filt)
  filt_mag = tf.clip_by_value(filt_mag, near_all_pass_alpha,
                              1.0 / near_all_pass_alpha)
  return tf.cast(filt_mag, tf.complex64) * tf.math.exp(
      1j * tf.cast(filt_phase, tf.complex64))


def double_stft(signal: tf.Tensor, frame_length: int, frame_step: int,
                meta_frame_length: int, meta_frame_step: int,
                nfft: Optional[int] = None, meta_nfft: Optional[int] = None,
                window_fn: Callable[..., tf.Tensor] = _sqrt_hann) -> tf.Tensor:
  """Returns double STFT of the input signal.

  A double STFT is an STFT followed by another STFT in the frames dimension
  of the first STFT. The first STFT output has (..., t1, f1) dimensions and
  then we transpose and obtain a (..., f1, t1) shape tensor and then we
  take another complex signal STFT on the t1 dimension and we get a
  (..., f1, t2, f2) dimensional tensor which we call the "double STFT" of the
  original signal.

  Note the STFTs are of signals which were padded in both begin and end parts.

  Args:
    signal: input signal of shape (..., t).
    frame_length: Frame length for the first STFT.
    frame_step: Frame shift for the first STFT.
    meta_frame_length: Frame length for the second STFT.
    meta_frame_step: Frame shift for the second STFT.
    nfft: Number of fft bins for the first STFT.
    meta_nfft: NUmber of fft bins for the second STFT.
    window_fn: Window function to use for both STFTs.
  Returns:
    signal_dstft: Double STFT of signal, has shape (..., f1, t2, f2) where f2
      is the meta frequency axis and f1 is the first STFT's frequency and
      t2 is meta-frames index after the second STFT.
  """
  signal_stft = _stft(signal, frame_length, frame_step, nfft, window_fn)
  signal_ft = tf.einsum('...tf->...ft', signal_stft)
  signal_ft_padded = _pad_beginning(signal_ft,
                                    meta_frame_length - meta_frame_step)
  signal_ftt = tf.signal.frame(signal_ft_padded, meta_frame_length,
                               meta_frame_step, pad_end=True)
  meta_window = tf.cast(window_fn(meta_frame_length), tf.complex64)
  signal_ftt_windowed = signal_ftt * meta_window
  if meta_nfft and meta_nfft > meta_frame_length:
    signal_ftt_windowed = _pad_to_length(signal_ftt_windowed, meta_nfft)
  return tf.signal.fft(signal_ftt_windowed)  # shape is (..., f_1, t_2, f_2)


def inverse_double_stft(signal_dstft: tf.Tensor, frame_length: int,
                        frame_step: int, meta_frame_length: int,
                        meta_frame_step: int, nfft: Optional[int] = None,
                        pad_begin: bool = True,
                        forward_window_fn: Callable[..., Any] = _sqrt_hann,
                        output_len: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Inverse double STFT."""
  signal_ftt = tf.signal.ifft(signal_dstft)
  i_window = tf.signal.inverse_stft_window_fn(
      meta_frame_step, forward_window_fn=forward_window_fn)(
          meta_frame_length, dtype=tf.float32)
  i_window = tf.cast(i_window, tf.complex64)
  signal_ftt = i_window * signal_ftt[..., :meta_frame_length]
  signal_ft = tf.signal.overlap_and_add(signal_ftt, meta_frame_step)
  if pad_begin:
    signal_ft = _clip_beginning(signal_ft, meta_frame_length - meta_frame_step)
  signal_tf = tf.einsum('...ft->...tf', signal_ft)
  return _istft(signal_tf, frame_length, frame_step, nfft, forward_window_fn,
                output_len=output_len)


def filter_in_double_stft(
    input_signal: tf.Tensor, target_signal: tf.Tensor, frame_length: int = 256,
    frame_step: int = 16, meta_frame_length: int = 256,
    meta_frame_step: int = 32, nfft: Optional[int] = None,
    meta_nfft: Optional[int] = None, frame_skip: int = 10,
    near_all_pass_alpha: float = 0.7) -> Tuple[tf.Tensor, tf.Tensor]:
  """Applies the best scalar complex filter in the double STFT domain."""
  input_len = _smart_dim(input_signal, -1)
  input_ds = double_stft(input_signal, frame_length, frame_step,
                         meta_frame_length, meta_frame_step, nfft=nfft,
                         meta_nfft=meta_nfft)
  target_ds = double_stft(target_signal, frame_length, frame_step,
                          meta_frame_length, meta_frame_step, nfft=nfft,
                          meta_nfft=meta_nfft)
  filt = calculate_scalar_filter(input_ds, target_ds, frame_skip=frame_skip)
  if near_all_pass_alpha > 0.0 and near_all_pass_alpha <= 1.0:
    filt = nap_project_scalar_filter(filt, near_all_pass_alpha)
  filtered_input_ds = filt * input_ds
  filtered_input_signal = inverse_double_stft(
      filtered_input_ds, frame_length, frame_step, meta_frame_length,
      meta_frame_step, nfft=nfft, output_len=input_len)
  return filtered_input_signal, filt


def filter_in_single_frame_stft(
    input_signal: tf.Tensor, target_signal: tf.Tensor, frame_length: int = 1024,
    frame_step: int = 256, nfft: Optional[int] = None,
    near_all_pass_alpha: float = 0.7, frame_skip: int = 1,
    window_fn: Callable[..., Any] = _sqrt_hann,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Filters input by a scalar to get close to target in single-frame STFT."""
  signal_len = _smart_dim(input_signal, -1)
  input_stft = _stft(input_signal, frame_length, frame_step, nfft, window_fn)
  target_stft = _stft(target_signal, frame_length, frame_step, nfft, window_fn)
  filt = calculate_scalar_filter(input_stft, target_stft, frame_skip=frame_skip)
  if near_all_pass_alpha > 0.0 and near_all_pass_alpha <= 1.0:
    filt = nap_project_scalar_filter(filt, near_all_pass_alpha)
  filtered_input_stft = filt * input_stft
  return _istft(filtered_input_stft, frame_length, frame_step, nfft, window_fn,
                output_len=signal_len), filt


def filter_in_multi_frame_stft(
    input_signal: tf.Tensor, target_signal: tf.Tensor, frame_length: int = 256,
    frame_step: int = 64, nfft: Optional[int] = None,
    near_all_pass_alpha: float = 0.7,
    near_all_pass_iterations: int = 1, filter_support_start: int = 0,
    filter_len: int = 10, frame_skip: int = 1, allow_scaling: bool = False,
    window_fn: Callable[..., Any] = _sqrt_hann,
    diagload: float = 1e-1, solver: str = 'default', fix_nans: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Filters input to get close to target in multi-frame STFT.

  Args:
    input_signal: Signal to be filtered, of shape (..., signal_len).
    target_signal: Signal to get close to, of shape (..., signal_len).
    frame_length: STFT frame length.
    frame_step: STFT frame step.
    nfft: STFT nfft size.
    near_all_pass_alpha: Magnitude FFT of the filter is constrained to be
      between alpha and 1/alpha.
    near_all_pass_iterations: Project this many times.
    filter_support_start: Filter support starting index. For causal filters,
      this is zero. For noncausal ones, this should be negative.
    filter_len: Filter length to use.
    frame_skip: Frames to skip when calculating the filter from the begin and
      end.
    allow_scaling: Allow overall scaling of the input_signal before near all
      pass projection.
    window_fn: Window function to use in the STFT.
    diagload: Diagonal loading with this value for solving the Toeplitz system.
    solver: Matrix solver type: 'default', 'lstsq', 'ls_cholesky', 'cholesky'.
    fix_nans: If the filter calculated has NaNs, replace it with a default
      identity (delta function) filter.
  Returns:
    filtered_input: Filtered input signal in time domain, gets close to target.
    filt: The multi-tap filter calculated in the STFT domain.
  """
  signal_len = _smart_dim(input_signal, -1)
  input_stft = _stft(input_signal, frame_length, frame_step, nfft, window_fn)
  target_stft = _stft(target_signal, frame_length, frame_step, nfft, window_fn)
  # Swap last two axes.
  input_stft = tf.einsum('...tf->...ft', input_stft)
  target_stft = tf.einsum('...tf->...ft', target_stft)
  filt = calculate_multitap_filter(input_stft, target_stft,
                                   frame_skip=frame_skip,
                                   filter_support_start=filter_support_start,
                                   filter_len=filter_len, diagload=diagload,
                                   solver=solver, fix_nans=fix_nans)
  if near_all_pass_alpha > 0.0 and near_all_pass_alpha <= 1.0:
    filt = nap_project_multitap_filter(filt,
                                       allow_scaling=allow_scaling,
                                       alpha=near_all_pass_alpha,
                                       iterations=near_all_pass_iterations)
  filtered_input_stft = perform_filtering(input_stft, filt,
                                          filter_support_start)
  # Swap back last two axes.
  filtered_input_stft = tf.einsum('...ft->...tf', filtered_input_stft)
  filtered_input_signal = _istft(filtered_input_stft, frame_length, frame_step,
                                 nfft, window_fn, output_len=signal_len)
  return filtered_input_signal, filt


def filter_in_time_domain(input_signal: tf.Tensor, target_signal: tf.Tensor,
                          filter_support_start: int = 0,
                          filter_len: int = 100,
                          near_all_pass_alpha: float = 0.0,
                          near_all_pass_iterations: int = 1,
                          allow_scaling: bool = False,
                          diagload: float = 0.1,
                          solver: str = 'default',
                          fix_nans: bool = False,
                          ) -> Tuple[tf.Tensor, tf.Tensor]:
  """Filters input_signal to get close to target_signal in the time domain.

  This is the same as projecting target_signal onto a subspace spanned by
  shifted versions of the input_signal. The subspace projected is the one
  obtained as the span obtained by delayed versions of the input_signal,
  with delays determined by filter support.

  Args:
    input_signal: Input signal with shape (..., signal_len)
    target_signal: Target signal with shape (..., signal_len)
    filter_support_start: Filter support starting index.
    filter_len: Filter length.
    near_all_pass_alpha: Magnitude response of the filter is clipped to the
      range (alpha, 1/alpha) if nonzero.
    near_all_pass_iterations: Project this many times.
    allow_scaling: If True, allow an overall scaling of the filter by the mean
      absolute value of the non-projected filters in the batch.
    diagload: Diagonal loading with this value for solving the Toeplitz system.
    solver: Matrix solver type: 'default', 'lstsq', 'ls_cholesky', 'cholesky'.
    fix_nans: If the filter calculated has NaNs, replace it with a default
      identity (delta function) filter.
  Returns:
    filtered_input_signal: Target signal projected onto the subspace spanned
      by delayed versions of the input signal. This signal can be used
      as a new (filtered) input signal in further processing.
    filt: The filter that acts on the input signal.
  """
  filt = calculate_multitap_filter(input_signal, target_signal,
                                   filter_support_start=filter_support_start,
                                   filter_len=filter_len,
                                   solver=solver,
                                   diagload=diagload,
                                   fix_nans=fix_nans)
  if near_all_pass_alpha > 0.0 and near_all_pass_alpha <= 1.0:
    filt = nap_project_multitap_filter(filt,
                                       allow_scaling=allow_scaling,
                                       alpha=near_all_pass_alpha,
                                       iterations=near_all_pass_iterations)
  # Filtering.
  filtered_input_signal = perform_filtering(input_signal, filt,
                                            filter_support_start)
  return filtered_input_signal, filt


def filter_with_best_filter(input_signal: tf.Tensor,
                            target_signal: tf.Tensor,
                            method: str = 'time_domain',
                            filter_support_start: int = 0,
                            filter_len: int = 100,
                            near_all_pass_alpha: float = 0.0,
                            allow_scaling: bool = False,
                            diagload: float = 0.1,
                            solver: str = 'default',
                            min_norm_ratio_for_multi_tap: float = 0.001,
                            fix_nans: bool = False,
                            frame_length: int = 256,
                            frame_step: int = 32,
                            ) -> tf.Tensor:
  """Finds and applies the best filter that gets input closest to target.

  Args:
    input_signal: Input signal (..., signal_len).
    target_signal: Target signal (..., signal_len).
    method: 'time_domain', 'single_frame_stft', 'multi_frame_stft', or
      'double_stft'.
      'time_domain': Finds a filter in time domain by solving a multi-tap
        filter prediction problem using a Toeplitz matrix solve.
      'single_frame_stft': Finds a single-tap complex filter in
        STFT domain by choosing a large enough frame_length to cover the
        desired filter_len, ignores filter_support_start.
      'multi_frame_stft': Finds a multi-tap complex filter in the STFT
        domain. It solves a Toeplitz problem in the STFT domain with complex
        numbers. It uses the frame_length and frame_step provided to perform
        an STFT and finds how many frames are required to cover the desired
        filter_len.
      'double_stft': Finds a single-tap complex filter in double STFT domain. A
        double STFT is the STFT of an STFT. It uses the provided
        frame_length and frame_step to perform its first STFT. Then it
        chooses meta_frame_length to cover the desired filter_len for the
        second STFT. It also ignores filter_support_start.
    filter_support_start: Starting index for the filter support.
    filter_len: Filter length in time domain.
    near_all_pass_alpha: Magnitude response of the filter is constrained to be
      between near_all_pass_alpha and 1.0 / near_all_pass_alpha
    allow_scaling: If True, allow arbitrary scaling of the filter.
    diagload: Diagonal loading for multi-tap filter solutions.
    solver: Solver for multi-tap filter solutions.
    min_norm_ratio_for_multi_tap: If the energy ratio between at least one of
      the input and target signals is lower than this value or higher than
      its reciprocal, we change multi-tap filtering to single-tap STFT domain
      filtering for numerical stability. Note that multi-tap filtering is done
      when method is 'time_domain' or 'multi_frame_stft'.
    fix_nans: If True, replace NaN filters with the identity filter in multi-tap
      filter solutions, i.e. when method is 'time_domain' or 'multi_frame_stft'.
    frame_length: Frame length for multi_frame_stft and double_stft. Note that
      single_frame_stft uses a frame_length that is large enough to cover the
      desired filter_len as well as a proportional frame_step, so it does not
      allow a separate frame_length or frame_step arguments.
    frame_step: Frame step for multi_frame_stft and double_stft.
  Returns:
    filtered_input_signal: Filtered input signal (..., signal_len).
  """
  if (min_norm_ratio_for_multi_tap > 0. and
      (method == 'time_domain' or method == 'multi_frame_stft')):
    norm_ratio = (tf.linalg.norm(input_signal, ord='euclidean', axis=-1) /
                  tf.linalg.norm(target_signal, ord='euclidean', axis=-1))
    safe_to_continue = tf.math.reduce_all(tf.logical_and(
        norm_ratio > min_norm_ratio_for_multi_tap,
        norm_ratio < 1.0 / min_norm_ratio_for_multi_tap))
    fn_original = functools.partial(filter_with_best_filter,
                                    method=method,
                                    filter_support_start=filter_support_start,
                                    filter_len=filter_len,
                                    near_all_pass_alpha=near_all_pass_alpha,
                                    allow_scaling=allow_scaling,
                                    diagload=diagload,
                                    solver=solver,
                                    min_norm_ratio_for_multi_tap=0.,
                                    fix_nans=fix_nans)
    # We change the method to 'single_frame_stft' which is a safer option since
    # no matrix solve is required.
    fn_modified = functools.partial(filter_with_best_filter,
                                    method='single_frame_stft',
                                    filter_support_start=filter_support_start,
                                    filter_len=filter_len,
                                    near_all_pass_alpha=near_all_pass_alpha,
                                    allow_scaling=allow_scaling,
                                    diagload=diagload,
                                    solver=solver,
                                    min_norm_ratio_for_multi_tap=0.,
                                    fix_nans=fix_nans)
    return tf.cond(safe_to_continue,
                   lambda: fn_original(input_signal, target_signal),
                   lambda: fn_modified(input_signal, target_signal))

  if method == 'time_domain':
    filt_signal, _ = filter_in_time_domain(
        input_signal, target_signal,
        filter_support_start=filter_support_start,
        filter_len=filter_len, near_all_pass_alpha=near_all_pass_alpha,
        allow_scaling=allow_scaling, solver=solver, fix_nans=fix_nans,
        diagload=diagload)

  elif method == 'multi_frame_stft':
    filter_len = -(-filter_len // frame_step)
    filter_support_start = filter_support_start // frame_step
    filt_signal, _ = filter_in_multi_frame_stft(
        input_signal, target_signal, frame_length=frame_length,
        frame_step=frame_step, filter_len=filter_len,
        filter_support_start=filter_support_start,
        near_all_pass_alpha=near_all_pass_alpha, solver=solver,
        fix_nans=fix_nans, diagload=diagload)

  elif method == 'single_frame_stft':
    # Note that single_frame_stft ignores filter_support_start.
    frame_length = int(2 * 2**np.ceil(np.log2(filter_len)))
    frame_step = frame_length // 4
    filt_signal, _ = filter_in_single_frame_stft(
        input_signal, target_signal, frame_length=frame_length,
        frame_step=frame_step, near_all_pass_alpha=near_all_pass_alpha)

  elif method == 'double_stft':
    # Note that double_stft ignores filter_support_start and finds a centered
    # filter, that is as if filter_support_start = - filter_len // 2.
    meta_frame_step = int(2.**np.ceil(np.log2(-(-filter_len // frame_step))))
    meta_frame_length = 8 * meta_frame_step
    filt_signal, _ = filter_in_double_stft(
        input_signal, target_signal, frame_length=frame_length,
        frame_step=frame_step, meta_frame_length=meta_frame_length,
        meta_frame_step=meta_frame_step,
        near_all_pass_alpha=near_all_pass_alpha, frame_skip=2)

  else:
    raise ValueError('Unknown filtering method {method}.')

  return filt_signal



