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

r"""Makes lists of background and foreground source files."""
import argparse
import glob
import os
import soundfile as sf


def make_lists(fsd_dir):
  """Makes background and foreground source lists under fsd_dir subsets."""
  subsets = ['train', 'validation', 'eval']
  short_cutoff = 10.0

  for subset in subsets:
    long_file_list = []
    short_file_list = []

    file_list = glob.glob(os.path.join(fsd_dir, subset, '*', '*.wav'))

    for myfile in file_list:
      relative_path = os.path.relpath(myfile, fsd_dir)

      file_info = sf.info(myfile)

      if file_info.duration > short_cutoff:
        long_file_list.append(relative_path)
      else:
        short_file_list.append(relative_path)

    n_short = len(short_file_list)
    if n_short > 0:
      list_name = os.path.join(fsd_dir, subset + '_foreground.txt')
      with open(list_name, 'w') as f:
        f.writelines('\n'.join(short_file_list))
      print('Generated foreground file list of {} files '
            'for {}.'.format(n_short, subset))

    n_long = len(long_file_list)
    if n_long > 0:
      list_name = os.path.join(fsd_dir, subset + '_background.txt')
      with open(list_name, 'w') as f:
        f.writelines('\n'.join(long_file_list))
      print('Generated background file list of {} files '
            'for {}.'.format(n_long, subset))


def main():
  parser = argparse.ArgumentParser(
      description='Makes background and foreground source lists.')
  parser.add_argument(
      '-d', '--data_dir', help='FSD data main directory.', required=True)
  args = parser.parse_args()
  make_lists(args.data_dir)

if __name__ == '__main__':
  main()
