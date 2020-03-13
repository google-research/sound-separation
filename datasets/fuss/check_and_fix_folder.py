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
r"""A script to check source mixtures.

Usage example:
python3 check_folder.py -sd /data/dcase2020_task4_ss_dev
-
"""
import argparse
import re

from utils import check_and_correct_example
from utils import make_example_list_from_folder


def check_folder(root_dir, ss_regex=re.compile(r'example.*_sources'),
                 check_length=True, fix_length=True,
                 check_mix=True, fix_mix=True, sample_rate=16000,
                 duration=10.0):
  """Check consistency of mixtures in folder.

  Args:
    root_dir: Output folder name where reverberated sources are written into.
    ss_regex: Regex for source file folders that contain sources.
    check_length: If True, check length.
    fix_length: If True, fix length problems.
    check_mix: If True, check mixture consistency.
    fix_mix: If True, fix mixture consistency.
    sample_rate: Sample rate.
    duration: Duration of wavs in seconds.
  """
  example_list = make_example_list_from_folder(root_dir, subset='all',
                                               ss_regex=ss_regex,
                                               pattern='_sources',
                                               subfolder_events=None)
  check_list(example_list, root_dir, check_length, fix_length,
             check_mix, fix_mix, sample_rate=sample_rate, duration=duration)


def check_list(example_list, root_dir, check_length=True, fix_length=True,
               check_mix=True, fix_mix=True, sample_rate=16000, duration=10.0):
  num_examples = len(example_list)
  print('Starting to check {} examples.'.format(num_examples))
  length_problem = 0
  fixed_length = 0
  mix_problem = 0
  fixed_mix = 0
  for example in example_list:
    lp, fl, mp, fm = check_and_correct_example(
        example, root_dir, check_length, fix_length, check_mix, fix_mix,
        sample_rate=sample_rate, duration=duration)
    length_problem += lp
    fixed_length += fl
    mix_problem += mp
    fixed_mix += fm
  print('Finished checking {} examples.'.format(num_examples))
  print('{} source and mixture files had length problems, {} of them '
        'fixed.'.format(length_problem, fixed_length))
  print('{} examples out of {} had mixture consistency problems, {} of them '
        'fixed.'.format(mix_problem, num_examples, fixed_mix))


def main():
  parser = argparse.ArgumentParser(
      description='Checks and fixes a scaper produced directory for length '
      'consistency and mixture consistency problems.')
  parser.add_argument(
      '-sd', '--source_dir', help='Source directory.', required=True)
  parser.add_argument(
      '-sl', '--source_list_file', help='Tab separated source list file.',
      required=False)
  args = parser.parse_args()
  if args.source_list_file:
    with open(args.source_list_file, 'r') as f:
      example_list = f.read().splitlines()
    check_list(example_list, args.source_dir)
  else:
    check_folder(args.source_dir)


if __name__ == '__main__':
  main()
