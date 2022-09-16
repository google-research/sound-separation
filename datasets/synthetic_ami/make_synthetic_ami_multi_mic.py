# Copyright 2022 Google LLC
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
r"""A script to create synthetic multi-channel AMI test set.

Usage example:
python3 make_synthetic_ami_multi_mic.py -a ${AMI_DIRECTORY} -o
${OUTPUT_DIRECTORY}
"""
import argparse
import os

import make_synthetic_ami_lib
import pandas as pd

NUM_MICS = 8


def main():
  parser = argparse.ArgumentParser(
      description='Creates synthetic AMI test set.\n Required arguments:\n')
  parser.add_argument(
      '-a', '--ami_directory', help='Path to downloaded AMI dataset.',
      required=True)
  parser.add_argument(
      '-o', '--output_directory', help='Path to output directory for wavs.',
      required=True)
  args = parser.parse_args()

  df = pd.read_csv(os.path.join(os.getcwd(), 'mixing_test.csv'))
  rows = df.iterrows()

  for i, row in rows:
    print(f'Creating example {i + 1} of {len(df)} total...')

    for j in range(NUM_MICS):
      # Create example.
      mic_name = f'Array1-0{j+1}'
      receiver_audio, source_images, source_audio = (
          make_synthetic_ami_lib.row_to_example_with_filtering(
              row, args.ami_directory, mic_name))
      # Write wav files.
      subdir = os.path.join(args.output_directory, '%010d' % i)
      if not os.path.exists(subdir):
        os.makedirs(subdir)
      make_synthetic_ami_lib.write_wav(
          os.path.join(subdir, f'receiver_audio_{j+1}.wav'),
          receiver_audio)
      make_synthetic_ami_lib.write_wav(
          os.path.join(subdir, f'source_images_{j+1}.wav'),
          source_images)
      make_synthetic_ami_lib.write_wav(
          os.path.join(subdir, f'source_audio_{j+1}.wav'), source_audio)


if __name__ == '__main__':
  main()
