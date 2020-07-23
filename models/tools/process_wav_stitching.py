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
r"""Process a .wav file in a stitching mode with a separation model.

The stitching mode can be used to process long-form audio where for example
we would like to use a 2-speaker separation model in a long meeting recording
containing more than 2 speakers but where we assume there are not more than 2
speakers active in a block_size window. So, that our 2-speaker separation model
can run for the whole meeting in a block-by-block fashion producing two tracks
containing non-overlapping speech regardless of the total number of speakers
in the meeting.

python3 process_wav_stitching \
--model_dir /tmp/mixable_sss_8mic_model/ \
--input /tmp/libricss_ov40.wav \
--output /tmp/libricss_ov40_sss_8mic_bf2_10s_processed.wav \
--block_size_in_seconds 10 --permutation_invariant True --window_type vorbis \
--input_channels 8 \
--output_tensor "model_iter_1/beamformed_waveforms:0" \
--write_outputs_separately True
"""
# pylint: enable=line-too-long

import argparse
import os
from typing import Tuple, Optional

import inference
import numpy as np
import stitching
import tensorflow.compat.v1 as tf


strtobool = inference.strtobool


def _extract_blocks_from_input(input_wav_file: str,
                               num_samples_in_block: int,
                               input_channels: int = 0,
                               scale_input: bool = False,
                               window_type: str = 'rectangular',
                               ) -> Tuple[np.ndarray, np.ndarray]:
  """Reads input wav file and extracts blocks from it.

  Args:
    input_wav_file: Input signal .wav file path.
    num_samples_in_block: Block size in samples.
    input_channels: If positive, truncate/extend the input signal to
      this number of channels, otherwise keep all channels at the input.
    scale_input: If True, scale input to have an absolute maximum of 0.99.
    window_type: Window type to use.
  Returns:
    input_blocks_np: Input signal in blocks, np.ndarray, with shape
      [num_blocks, num_mics, num_samples_in_block].
    input_len_np: Input signal length in samples, integer.
    sample_rate_np: Sample rate, integer.
  """
  hop_size_in_samples = num_samples_in_block // 2
  # Define the graph which extracts input blocks.
  graph_input = tf.Graph()
  with graph_input.as_default():
    input_wav, sample_rate = inference.read_wav_file(
        input_wav_file, input_channels, scale_input)
    input_wav = tf.transpose(input_wav)  # shape: [mics, samples]
    input_len = tf.shape(input_wav)[-1]
    # We pre-pad the input signal since we apply a window function and the
    # first block's first half only has a single window function in the
    # overlap-add reconstruction, so we pad it such that we can ignore the
    # first half after reconstruction by overlap-add.
    input_wav = tf.pad(input_wav, [[0, 0], [hop_size_in_samples, 0]])
    input_blocks = tf.signal.frame(input_wav,
                                   num_samples_in_block,
                                   hop_size_in_samples,
                                   pad_end=True)

    input_blocks *= stitching.get_window(window_type, num_samples_in_block)
    # Transpose to make blocks as batch items.
    input_blocks = tf.transpose(input_blocks, (1, 0, 2))
    # input_blocks has shape (batch/blocks, mics, samples_in_block)

  # First graph is used to extract the input blocks from the input wav file.
  with tf.Session(graph=graph_input) as sess:
    input_blocks_np, input_len_np, sample_rate_np = sess.run(
        [input_blocks, input_len, sample_rate])
  return input_blocks_np, input_len_np, sample_rate_np


def _run_model_for_blocks(input_blocks_np: np.ndarray,
                          model_dir: str,
                          checkpoint: Optional[str],
                          input_tensor_name: str,
                          output_tensor_name: str) -> np.ndarray:
  """Runs separation model for each block.

  The input is a multi-channel signal, but the output is a single channel
  output per source signal.

  Args:
    input_blocks_np: Input mixture signal samples, np.ndarray with shape
      [num_blocks, num_mics, num_samples_in_block].
    model_dir: Model directory with at least one checkpoint and inference.meta
      file.
    checkpoint: If not None, checkpoint path to use, otherwise use the
      latest checkpoint in the model_dir.
    input_tensor_name: The name of the input tensor in the model.
    output_tensor_name: The name of the output tensor in the model.
  Returns:
    output_blocks_np: Output signal samples, np.ndarray with shape
      [num_blocks, num_sources, num_samples_in_block].
  """

  model_graph_filename = os.path.join(model_dir, 'inference.meta')
  tf.logging.info('Importing meta graph: %s', model_graph_filename)

  if not checkpoint:
    checkpoint = tf.train.latest_checkpoint(model_dir)
  # Use separation model.
  separation_model = inference.SeparationModel(
      checkpoint, model_graph_filename, input_tensor_name,
      output_tensor_name)
  output_blocks = []
  for i in range(input_blocks_np.shape[0]):
    print('Processing block %d of %d...' % (i+1, input_blocks_np.shape[0]))
    output_blocks.append(separation_model.separate(input_blocks_np[i]))
  output_blocks_np = np.stack(output_blocks, axis=0)
  return output_blocks_np


def _resolve_permutation_and_write_output(
    output_wav_file: str, sample_rate: float,
    output_blocks_np: np.ndarray, input_len_np: np.ndarray,
    window_type: str, permutation_invariant: bool,
    output_channels: int, write_outputs_separately: bool):
  """Resolves permutation across blocks and writes output .wav files.

  Args:
    output_wav_file: Output .wav file path.
    sample_rate: Sampling rate for the output signals.
    output_blocks_np: Output signal in blocks, np.ndarray with shape
      [num_blocks, num_sources, num_samples_in_block].
    input_len_np: Input signal length in samples, so we can truncate the
      output(s) to this length when writing.
    window_type: Window type to use.
    permutation_invariant: If True, the model is trained with a
      permutation invariant objective, so the output order of sources
      are arbitrary.
    output_channels: If positive, the number of sources to output, otherwise
      output all sources.
    write_outputs_separately: If True, write output for each source in a
      separate file derived from the output_wav_file path, otherwise write
      them in a single multi-channel .wav file.
  Returns:
    Nothing, but writes the output signals into output path(s).
  """
  # Define a graph which resolves permutation if required and writes
  # output signals.
  num_samples_in_block = output_blocks_np.shape[-1]
  num_sources = output_blocks_np.shape[1]
  hop_samples = num_samples_in_block // 2
  graph_output = tf.Graph()
  with graph_output.as_default():
    window = stitching.get_window(window_type, num_samples_in_block)
    output_blocks_placeholder = tf.placeholder(
        tf.float32, shape=(None, num_sources, num_samples_in_block))
    input_len_placeholder = tf.placeholder(tf.int32, shape=())
    output_blocks = output_blocks_placeholder
    if permutation_invariant:
      output_blocks = stitching.sequentially_resolve_permutation(
          output_blocks, window)
    output_blocks = tf.transpose(output_blocks, (1, 0, 2))
    # output_blocks now has shape (sources, blocks, samples)
    # We apply the window twice since its overlap-added squared sum is 1.0.
    output_blocks *= window
    output_wavs = tf.signal.overlap_and_add(output_blocks, hop_samples)
    output_wavs = tf.transpose(output_wavs)
    # We ignore the padded first hop_samples samples.
    output_wavs = output_wavs[
        hop_samples: input_len_placeholder + hop_samples, :]

    write_output_ops = inference.write_wav_file(
        output_wav_file, output_wavs, sample_rate=sample_rate,
        num_channels=num_sources,
        output_channels=output_channels,
        write_outputs_separately=write_outputs_separately,
        channel_name='source')

  # The graph is used to resolve permutation across blocks if required,
  # and writes the output source signals.
  with tf.Session(graph=graph_output) as sess:
    sess.run(write_output_ops,
             feed_dict={output_blocks_placeholder: output_blocks_np,
                        input_len_placeholder: input_len_np})


def main():
  parser = argparse.ArgumentParser(
      description='Process a long mixture .wav file to separate into sources '
      'by using block processing and combining block outputs through '
      'stitching.')
  parser.add_argument(
      '-i', '--input', help='Input .wav file.', required=True, type=str)
  parser.add_argument(
      '-o', '--output', help='Output .wav file.', required=True, type=str)
  parser.add_argument(
      '-m', '--model_dir', help='Model root directory, required. '
      'Must contain inference.meta and at least one checkpoint.', type=str)
  parser.add_argument(
      '-ic', '--input_channels', help='Truncate/pad input to this many '
      'channels if positive.',
      default=0, type=int)
  parser.add_argument(
      '-oc', '--output_channels', help='Limit the number of output sources to '
      'this number, if positive.', default=0, type=int)
  parser.add_argument(
      '-it', '--input_tensor', default='input_audio/receiver_audio:0',
      help='Name of tensor to which to feed input_wav.', type=str)
  parser.add_argument(
      '-ot', '--output_tensor', default='denoised_waveforms:0',
      help='Name of tensor to output as output_wav.', type=str)
  parser.add_argument(
      '-wos', '--write_outputs_separately', default=True,
      help='Write output source signals into separate wav files.',
      type=strtobool)
  parser.add_argument(
      '-wt', '--window_type', default='rectangular', type=str,
      help='Window type: rectangular, vorbis or kaiser-bessel-derived.')
  parser.add_argument(
      '-bs', '--block_size_in_seconds', default=10.0, type=float,
      help='Block size used for stitching processing.')
  parser.add_argument(
      '-sr', '--sample_rate', default=16000, help='Sample rate.', type=int)
  parser.add_argument(
      '-pi', '--permutation_invariant', default=False, type=strtobool,
      help='If True, perform permutation invariant stitching.')
  parser.add_argument(
      '-si', '--scale_input', default=False, help='If True, scale the input '
      'signal such that its absolute maximum value is 0.99.', type=strtobool)
  parser.add_argument(
      '-c', '--checkpoint', default=None, help='Override for checkpoint path.')
  args = parser.parse_args()
  output_dir = os.path.dirname(args.output)
  os.makedirs(output_dir, exist_ok=True)

  # We run three tf sessions with three different graphs.
  # TODO(user): In the future, we may find a way to run the whole
  # process as a single tensorflow graph.
  # To make it work, either (1) we would need to be able to run the inference
  # graph in batch mode with a dynamic batch size, or (2) we should be able to
  # import a graph and convert it to a tf function and
  # sequentially obtain each block output from a block input in tensorflow
  # using a while loop or similar graph looping construct. I tried but neither
  # of these approaches worked for me, so we run three sessions.

  # Make sure there are even number of samples in each block.
  block_size_in_samples = 2 * int(
      round(args.block_size_in_seconds * float(args.sample_rate) / 2.0))
  input_blocks_np, input_len_np, sample_rate = _extract_blocks_from_input(
      args.input, block_size_in_samples, args.input_channels,
      args.scale_input, args.window_type)

  assert sample_rate == args.sample_rate

  output_blocks_np = _run_model_for_blocks(
      input_blocks_np, args.model_dir, args.checkpoint, args.input_tensor,
      args.output_tensor)

  _resolve_permutation_and_write_output(
      args.output, sample_rate, output_blocks_np, input_len_np,
      args.window_type, args.permutation_invariant,
      args.output_channels, args.write_outputs_separately)


if __name__ == '__main__':
  main()
