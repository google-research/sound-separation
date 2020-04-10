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
r"""Converts a mixture / source lists from DESED format to ssdata format.

Usage example:
SCRIPT_PATH=/data/src && \
LISTS_PATH=/data/lists && \
BASE_OUT=DESED_synthetic_2020_audio_train_synthetic20_soundscapes && \
python3 ${SCRIPT_PATH}/convert_desed_list.py \
--mixtures ${LISTS_PATH}/${BASE_OUT}_train.txt \
--sources ${LISTS_PATH}/${BASE_OUT}_sources_train.txt \
--output ${LISTS_PATH}/${BASE_OUT}_sslist_train.txt
"""
import argparse
import collections


def main():
  parser = argparse.ArgumentParser(
      description='Make mixing list.')
  parser.add_argument(
      '-m', '--mixtures', help='DESED mixture list', required=True)
  parser.add_argument(
      '-s', '--sources', help='desed mixture list', required=True)
  parser.add_argument(
      '-o', '--output', help='Output list file.', required=True)
  args = parser.parse_args()

  mixtures = args.mixtures
  sources = args.sources
  with open(sources, 'r') as f:
    sourcelines = f.read().splitlines()
  with open(mixtures, 'r') as f:
    mixlines = f.read().splitlines()

  key_list = []
  source_dict = collections.defaultdict(list)
  for line in mixlines:
    key = line.split('.')[0]
    source_dict[key].append(line)
    key_list.append(key)
  for line in sourcelines:
    key = line.split('_')[0]
    source_dict[key].append(line)
  with open(args.output, 'w') as f:
    for key in key_list:
      line = source_dict[key]
      f.write('\t'.join(line) + '\n')

if __name__ == '__main__':
  main()
