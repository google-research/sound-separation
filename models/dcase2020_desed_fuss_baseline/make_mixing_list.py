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
r"""Makes a sources list by combining across different datasets.

Usage example using local folders:
python3 make_mixing_list.py --out ${OUT_LIST} --dirs ${DIR1} ${DIR2} \
  --lists ${LIST1} ${LIST2} --num ${NUM_MIX} --random_seed ${RANDOM_SEED}

Usage example, to combine FUSS and DESED sources, combining over instances of
the same event in DESED, while renumbering the different instances in FUSS
to keep them distinct, and limiting FUSS sounds to 2, and DESED sounds to 3,
and choosing these sources at random from each mixture:

SCRIPT_DIR=/data/src && \
DATASET_DIR=/data && \
DESED_DIR=${DATASET_DIR}/DESED_synthetic_2020 && \
FUSS_DIR=${DATASET_DIR}/dcase2020/ssdata/ && \
python3 ${SCRIPT_DIR}/make_mixing_list.py \
  --out ${DESED_DIR}/FUSS_DESED_sources_list_randomize_test.txt \
  --dirs ${FUSS_DIR} ${DESED_DIR}/audio/train/synthetic20/soundscapes/ \
  --lists ${FUSS_DIR}/train_example_list.txt \
  ${DESED_DIR}/DESED_synthetic_2020_audio_train_synthetic20_sslist_train.txt \
  --modes 'sources' 'sources' --max_sources 2 3 --data_names FUSS DESED \
  --split_instances split no_split --num 10000 --randomize_sources \
  --random_seed 2020

"""
import argparse
import collections
import os
import re

import numpy as np


def split_class_instances(classnames):
  """Split sounds from the same class into separate classes.

  After splitting, we add a count number after instance number 1.
  For example original class names of ['cat', 'dog', 'cat'] will be returned
  as ['cat', 'dog', 'cat_2'].

  Args:
    classnames: list of classes.
  Returns:
    list of modified classnames.
  """
  classname_count = collections.defaultdict(int)
  new_classnames = []
  for classname in classnames:
    classname_count[classname] += 1
    if classname_count[classname] > 1:
      new_classnames.append(classname +
                            '_{}'.format(classname_count[classname]))
    else:
      new_classnames.append(classname)
  return new_classnames


def wav_to_class(filename, data_name, mymode,
                 background_is_separate_class=True):
  """Extract class from a filename or classes from a list of filenames.

  The class name is a combination of letters and '_'.
  A prefix ending in digit + '_' is discarded.
  A suffix of _nOn, _nOff, or _nOn_nOff, found in DESED data, is also discarded.
  For example, if filename is:
    '/thing1/thing2/foreground4_Electric_shaver_toothbrush_nOff.wav'
  then the class name is:
    'Electric_shaver_toothbrush'.

  Args:
    filename: full path to wav file
      (path and extension are ignored if present).
      Special filename of '0' indicates a missing (all-zero) source.
      Or a list of filenames.
    data_name: name to use for dataset, such as 'FUSS' or 'DESED', which is
      padded in front of the class name to yield a classname like
      'DESED_cat' and 'FUSS_sound'.
    mymode: 'mixture', 'sources' or 'one_source'.
      When mymode == 'mixture', the class id returned as
     'FUSS_mixture' or 'DESED_mixture'.
     background_is_separate_class: If True, background source is always a
      separate class by adding bg_ in front of the class name.
  Returns:
    class name.
  """

  if filename == '0':
    return '0'

  if mymode == 'mixture':
    return '{}_mixture'.format(data_name)
  else:
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    regexp = re.compile(r'.*ground[0-9]+_(?P<class>[a-zA-Z_]+)')
    match = regexp.search(base)
    if match:
      classname = match.group('class')
      classname = re.sub('_nOff$', '', classname)
      classname = re.sub('_nOn$', '', classname)
      if 'background' in base and background_is_separate_class:
        classname = 'bg_' + classname
      return '{}_{}'.format(data_name, classname)
    else:
      raise ValueError('Unknown source class type for file {}'.format(
          filename))


def class_map_type(s):
  """Argparse type for specifying class mappings."""
  class_maps = collections.OrderedDict()
  try:
    mapstrings = s.split(';')
    for mymap in mapstrings:
      map_from, map_to = mymap.split(',')
      class_maps[map_from] = map_to
    print('Class_maps: ', class_maps)
    return class_maps
  except:
    raise argparse.ArgumentTypeError(
        'Class maps must be like a,b;c,d but is {}'.format(s))


def main():
  parser = argparse.ArgumentParser(
      description='Makes a sources list by combining different datasets.'
      'Arguments to --lists, --dirs, and --modes must be the same length.')
  parser.add_argument(
      '-l', '--lists', nargs='+', help='List of input lists.', required=True)
  parser.add_argument(
      '-d', '--dirs', nargs='+', help='List of top level directories,'
      ' of same length as --lists.', required=True)
  parser.add_argument(
      '-m', '--modes', nargs='+', help='Mode of processing for each input.'
      'Same length as --lists. '
      'Modes are: mixture (use the mixture), sources (use all sources), '
      ' one_source (use a singlesource).', required=False)
  parser.add_argument(
      '-ms', '--max_sources', nargs='+',
      help='Maximum number of sources to use to form a mixture.'
      'Same length as --lists. ', type=int, required=False)
  parser.add_argument(
      '-dn', '--data_names', nargs='+', help='List of data names to be used in '
      ' classes for each input.  Same length as --lists. ', required=False)
  parser.add_argument(
      '-si', '--split_instances', nargs='+', help='List of instance splitting '
      'options for each input.  Same length as --lists. Each option can be: '
      'split (number the instances of the same class), no_split (leave the '
      'class names the same).', required=False)
  parser.add_argument(
      '-o', '--out', help='Output list file.', required=True)
  parser.add_argument(
      '-n', '--num', help='Number of random mixtures to generate.',
      required=False, default=100, type=int)
  parser.add_argument(
      '-rsrc', '--randomize_sources', help='Choose a random subset of sources '
      'from each mixture (only applies to the mode sources)', required=False,
      dest='randomize_sources', action='store_true')
  parser.set_defaults(randomize_sources=False)
  parser.add_argument(
      '-rseed', '--random_seed', help='Random seed.', required=False,
      default=123, type=int)
  parser.add_argument('-cm', '--class_maps',
                      help=r'''Class maps in the format a,b;c,d which specifies'
                      'mapping a -> b and c -> d. These maps '
                      'are applied sequentially in the order they are given '
                      'in the command line argument and they can contain '
                      'regular expressions.  Note, the regular expression must'
                      'match the whole class name, and you can use a group to '
                      'identify the part to be kept (e.g., use '
                      '"hi(.*)" -> "bye\g<1>" to convert from '
                      '"hi_dave" to "bye_dave"''',
                      type=class_map_type)
  args = parser.parse_args()

  np.random.seed(args.random_seed)

  dirs = args.dirs
  lists = args.lists
  modes = args.modes
  max_sources = args.max_sources
  data_names = args.data_names
  split_instances = args.split_instances
  randomize_sources = args.randomize_sources
  class_maps = args.class_maps

  if modes is None:
    modes = ['mixture'] * len(lists)
  if max_sources is None:
    max_sources = [100] * len(lists)  # Use a high default value to get all.
  if data_names is None:
    data_names = ['dataset_' + i for i in range(len(lists))]
  if split_instances is None:
    split_instances = ['no_split'] * len(lists)

  assert (len(dirs) == len(lists) == len(modes) == len(max_sources)
          == len(data_names) == len(split_instances)), \
      'Length of dirs, lists, modes and max_sources needs to be the same'

  entries = {}
  num_total = {}
  num_max_per_mix = {}
  list_args = zip(lists, dirs, modes, max_sources,
                  data_names, split_instances)
  for mylist, mydir, mymode, max_src, data_name, split in list_args:
    do_split = (split == 'split')
    entries[mylist] = []
    num_max_per_mix[mylist] = max_src
    with open(mylist, 'r') as f:
      lines = f.read().splitlines()
    for line in lines:
      wavs = line.split('\t')
      wavs = [os.path.join(mydir, w) for w in wavs]
      mixture = wavs[0]
      sources = wavs[1:]

      if mymode == 'mixture':
        entries[mylist].append(['{}:{}'.format(
            wav_to_class(mixture, data_name, mymode), mixture)])

      elif mymode == 'sources':
        classes = [wav_to_class(src, data_name, mymode) for src in sources]
        if do_split:
          classes = split_class_instances(classes)
        entries[mylist].append([
            '{}:{}'.format(cls, src) for src, cls in zip(sources, classes)])

      elif mymode == 'one_source':
        classes = [wav_to_class(src, data_name, mymode) for src in sources]
        if do_split:
          classes = split_class_instances(classes)
        for cls, src in zip(sources, classes):
          entries[mylist].append(['{}:{}'.format(cls, src)])
      else:
        assert False, 'Unsupported mode {}'.format(mymode)

  for key in entries:
    num_total[key] = len(entries[key])

  with open(args.out, 'w') as f:
    for mix_num in range(args.num):
      line = ['{:06d}'.format(mix_num)]
      for key in entries:
        # Randomly choose number of entries to use for the mixture (including
        # background if present)
        enum = mix_num % num_total[key]
        if enum == 0:
          np.random.shuffle(entries[key])
        entry_sources = entries[key][enum]
        num_available_sources = len(entry_sources)
        if randomize_sources:
          source_count = np.random.randint(1, num_max_per_mix[key])
          source_count = min(num_available_sources, source_count)
          line += list(np.random.choice(entry_sources, source_count,
                                        replace=False))
        else:
          source_count = min(num_available_sources, num_max_per_mix[key])
          line += entry_sources[0:source_count]
      if class_maps:
        for map_from, map_to in class_maps.items():
          def map_class(field, map_from, map_to):
            if ':' not in field:
              return field
            cls, wav = field.split(':')
            cls = re.sub('^' + map_from + '$', map_to, cls)
            return cls + ':' + wav
          line = [map_class(field, map_from, map_to) for field in line]
      f.write('\t'.join(line) + '\n')

if __name__ == '__main__':
  main()
