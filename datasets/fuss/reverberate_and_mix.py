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
r"""A script to reverberate sources.

Usage example:
python3 reverberate_sources.py -s ${MIX_DIR} -r ${RIR_DIR} \
  -o ${REVERB_MIX_DIR} --write_mix_info ${MIX_INFO} --write_rirs ${RIR_LIST} \
  --read_sources ${SRC_LIST} --random_seed ${RANDOM_SEED}
"""
import argparse
import os
import re
import shutil

import numpy as np
from utils import make_example_dict_from_folder
from utils import read_wav
from utils import write_wav


def multimic_convolve(src_data, rir_data, output_advance=0):
  """Performs convolution for single channel src and multi channel rir data.

  Args:
    src_data: Source signal with shape [src_len].
    rir_data: RIR signals with shape [rir_len, num_mics].
    output_advance: How many samples to advance (move forward in time)
      the output signals by.
  Returns:
    reverberated sources of shape [src_len, num_mics].
  """
  num_mics = np.shape(rir_data)[-1]
  out = []
  src_len = np.shape(src_data)[-1]
  out_range = np.arange(output_advance, output_advance+src_len)
  for i in range(num_mics):
    out.append(np.convolve(src_data, rir_data[:, i], 'full')[out_range])
  return np.stack(out, axis=-1)


def make_rir_dict_from_folder(folder_rir,
                              rir_regex=re.compile(r'rirs_.*')):
  """Makes an rirs hierarchical dictionary.

  Returns a dictionary which maps subfolder -> rir_key -> list of rir wavs.

  Args:
    folder_rir: Main path to an rir folder which contains train validation eval
      subfolders.
    rir_regex: A regex that matches rir subfolder names.
  Returns:
    A hierarchical dictionary as described above.
  """
  rir_dict = {}
  for subfolder in ['train', 'validation', 'eval']:
    rir_dict[subfolder] = {}
    rir_subfolder_path = os.path.join(folder_rir, subfolder)
    if os.path.isdir(rir_subfolder_path):
      rir_entries = os.listdir(rir_subfolder_path)
      rir_selected = sorted(list(filter(rir_regex.search, rir_entries)))
      for rir_example in rir_selected:
        rir_example_relpath = os.path.join(subfolder, rir_example)
        rir_example_path = os.path.join(rir_subfolder_path, rir_example)
        rir_dict[subfolder][rir_example_relpath] = []
        if os.path.isdir(rir_example_path):
          rir_wavs = sorted(list(filter(re.compile(r'.*\.wav').search,
                                        os.listdir(rir_example_path))))
          for rir_wav in rir_wavs:
            rir_wav_f = os.path.join(rir_example_path, rir_wav)
            rir_wav_f_rel = os.path.relpath(rir_wav_f, folder_rir)
            rir_dict[subfolder][rir_example_relpath].append(rir_wav_f_rel)
  return rir_dict


def make_mix_info(source_dict, rir_dict,
                  subfolders=('train', 'validation', 'eval')):
  mix_info = {}
  for subfolder in subfolders:
    if subfolder not in source_dict:
      continue
    mix_info = make_mix_info_subsources(mix_info,
                                        source_dict[subfolder],
                                        rir_dict[subfolder])
  return mix_info


def make_mix_info_subsources(mix_info, sub_source, sub_rir,
                             assign_rir_based_on_class=False,
                             repeatedly_use_rirs=False):
  """Make a flat dictionary of mix_info.

  Args:
    mix_info: An initial dictionary which could be empty or it could
      have entries that map: reverb_mix_file -> [list of sources, list of rirs]
      to be used to calculate the reverberated mix_file.
    sub_source: A dictionary: dry_mix_file -> list of source files
    sub_rir: A dictionary: rir_folder_name (or rir_key) -> list of rir files
    assign_rir_based_on_class: Assign the same rir to be used for the sources
      that have the same class.
    repeatedly_use_rirs: Use rirs repeatedly if there are not enough rirs to
      cover all sources in the mixture.
  Returns:
    An updated mix_info dictionary.
  """
  list_of_mixtures = sorted(sub_source.keys())
  list_of_rirs = sorted(sub_rir.keys())
  num_mix = len(list_of_mixtures)
  num_rir = len(list_of_rirs)
  print('{} mixtures and {} rooms'.format(num_mix, num_rir))
  if num_rir == 0 and num_mix != 0 or num_mix == 0 and num_rir != 0:
    raise ValueError('Cannot continue with {} mixtures '
                     'and {} rooms.'.format(num_mix, num_rir))
  for i, mix in enumerate(list_of_mixtures):
    if i % num_rir == 0:
      np.random.shuffle(list_of_rirs)
    rir_key = list_of_rirs[i % num_rir]
    sources = sub_source[mix]
    rirs = sub_rir[rir_key]
    ordered_sources = []
    ordered_rirs = []
    for subtype in ['background', 'foreground']:
      # We assign sources and rirs to a subtype based on their path strings.
      sources_sub = [source for source in sources if subtype in source]
      rir_sub = [rir for rir in rirs if subtype in rir]
      if assign_rir_based_on_class:
        rir_num = 0
        rir_for_source_class = {}
        for source in sources_sub:
          # The class of each source is determined by looking at the text
          # after .*ground[0-9]+_ and before the extension \..* (usually .wav)
          # e.g. foreground12_dog.wav, background0_cat.wav etc. are supported.
          source_class = re.match(r'.*ground[0-9]+_(.*)\..*', source).groups(0)
          if source_class not in rir_for_source_class:
            if rir_num < len(rir_sub):
              rir_for_source_class[source_class] = rir_sub[rir_num]
            else:
              if repeatedly_use_rirs:
                rir_num = 0
                rir_for_source_class[source_class] = rir_sub[rir_num]
              else:
                raise ValueError('Need more rirs to support sources.')
            rir_num += 1
          ordered_sources.append(source)
          ordered_rirs.append(rir_for_source_class[source_class])
      else:
        if len(rir_sub) < len(sources_sub):
          if repeatedly_use_rirs:
            while len(rir_sub) < len(sources_sub):
              rir_sub += rir_sub
          else:
            raise ValueError('Not enough rirs in room {} to '
                             'support all sources in {}'.format(rir_key, mix))
        ordered_sources.extend(sources_sub)
        ordered_rirs.extend(rir_sub[:len(sources_sub)])
    mix_info[mix] = [ordered_sources, ordered_rirs]
  return mix_info


def reverberate_and_mix(out_folder, sources_folder, rir_folder,
                        mix_info, scale_rirs=10.0,
                        part=0, nparts=8, num_mics=1, chat=True,
                        output_align='causal'):
  """Reverberate and mix sources.

  Args:
    out_folder: Output folder to write reverberated sources and mixtures.
    sources_folder: Sources folder to read sources from.
    rir_folder: RIR folder to read rirs from.
    mix_info: A dictionary : mix_file_name -> (sources, rirs)
      where sources and rirs are paired lists of relative paths to source
      and rir signal wav files used in reverberate and mix operation to be
      performed.
    scale_rirs: A value to scale the RIR signals (float).
    part: Integer value indicating which part of parallel jobs to run (int).
    nparts: Number of parts considered for parallel runs (int).
    num_mics: Number of mics to use at the output (int).
    chat: If True, display more messages (bool).
    output_align: Output signal alignment type.
      'causal: Uses causal RIR filtering with no additional shift. '
      'align_sources': Find the average peak index of RIR(s) corresponding '
      '  each source and advance each source with that index. This has an '
      '  effect of aligning each source with their non-reverberated version.'
  Returns:
    None, but writes reverberated sources and mixtures into files.
  """
  list_mix = sorted(mix_info.keys())
  list_len = len(list_mix)
  partsize = list_len // nparts
  assert part < nparts
  start = part * partsize
  end = list_len if part == nparts-1 else (part + 1) * partsize
  if start == end:
    raise ValueError('Not enough mixtures to generate. Part {} of {} to '
                     'generate a total of {} mixtures.'.format(
                         part, nparts, list_len))
  print('Reverberating and mixing from {} to {} '
        'out of {}.'.format(start, end, list_len))
  for mix in list_mix[start:end]:
    sources, rirs = mix_info[mix]
    mix_to_data = []
    rir_peak_delays = []
    max_src_len = -1
    if chat:
      print('--\n{} ='.format(mix))
    for source, rir in zip(sources, rirs):
      source_path = os.path.join(sources_folder, source)
      src_data, samplerate_src = read_wav(source_path, always_2d=True)
      rir_path = os.path.join(rir_folder, rir)
      rir_data, samplerate_rir = read_wav(rir_path, always_2d=True)
      assert samplerate_src == samplerate_rir
      # Pick channel 0 of src_data.
      src_data = src_data[:, 0]
      # Pick num_mics channels of rirs and scale them.
      if len(rir_data.shape) == 2:
        rir_mics = np.shape(rir_data)[1]
        if rir_mics < num_mics:
          raise ValueError(f'The rir {rir_path} has only {rir_mics} channel '
                           f'data where the specified num_mics={num_mics}')
        rir_data = rir_data[:, :num_mics]
      else:
        if num_mics > 1:
          raise ValueError(f'The rir {rir_path} has only single channel data '
                           f'but specified num_mics={num_mics}')
        rir_data = np.reshape(rir_data, [-1, 1])
      rir_data = scale_rirs * rir_data
      rir_len = len(rir_data[:, 0])
      src_len = len(src_data)
      rir_max = np.max(np.abs(rir_data))
      rir_peaks = np.argmax(np.abs(rir_data), axis=0)
      src_max = np.max(np.abs(src_data))
      max_src_len = np.maximum(src_len, max_src_len)
      if chat:
        print('+ {} [{}, {:1.2f}] * {} [{}, {:1.2f}, {}]'.format(
            source, src_len, src_max, rir, rir_len, rir_max, rir_peaks))
      mix_to_data.append([src_data, rir_data, source, rir, rir_peaks])
    mix_rev_sources = []
    rir_paths_used = []
    for data in mix_to_data:
      src_data, rir_data, source_relpath, rir_relpath, rir_peaks = data
      rir_paths_used.append(rir_relpath)
      src_len = len(src_data)
      if src_len < max_src_len:
        print('WARNING: original source data has {} samples '
              'for source file {}, zero padding '
              'to size {}.'.format(src_len, source_relpath, max_src_len))
        src_data = np.concatenate((src_data, np.zeros(
            max_src_len - src_len)), axis=0)
      if output_align == 'align_sources':
        output_advance = np.round(np.mean(np.asarray(
            rir_peaks))).astype(np.int32)
      elif output_align == 'causal':
        output_advance = 0
      else:
        raise ValueError(f'Unknown output_align={output_align}')
      if chat and output_advance != 0:
        print(f'Source {source_relpath} advanced by {output_advance} samples.')
      rev_src_data = multimic_convolve(src_data, rir_data,
                                       output_advance=output_advance)
      # Write reverberated source data.
      rev_src_path = os.path.join(out_folder, source_relpath)
      os.makedirs(os.path.dirname(rev_src_path), exist_ok=True)
      write_wav(rev_src_path, rev_src_data, samplerate_src)
      mix_rev_sources.append(rev_src_data)
    mixed_rev_data = np.sum(np.stack(mix_rev_sources, axis=0), axis=0)
    mix_wav_path = os.path.join(out_folder, mix)
    mix_wav_base = mix_wav_path.rstrip('.wav')
    write_wav(mix_wav_path, mixed_rev_data, samplerate_src)
    in_wav_path = os.path.join(sources_folder, mix)
    in_wav_base = in_wav_path.rstrip('.wav')
    if os.path.exists(in_wav_base + '.jams'):
      shutil.copyfile(in_wav_base + '.jams', mix_wav_base + '.jams')
    if os.path.exists(in_wav_base + '.txt'):
      with open(in_wav_base + '.txt', 'r') as f:
        lines = f.readlines()
      with open(mix_wav_base + '.txt', 'w') as f:
        f.write(''.join(lines))
        f.write('\nroom impulse responses used:\n{}'.format(
            '\n'.join(rir_paths_used)))


def write_mix_info(mix_info, info_file):
  """Prints the content of mix_info dictionary into file."""
  if os.path.isfile(info_file):
    raise ValueError(f'mix_info file {info_file} already exists. Please '
                     f'delete {info_file} and re-run.')
  with open(info_file, 'w') as f:
    for mix in mix_info:
      line = '{} = '.format(mix)
      sources, rirs = mix_info[mix]
      components = []
      for source, rir in zip(sources, rirs):
        components.append('{} * {}'.format(source, rir))
      line += (' + '.join(components) + '\n')
      f.write(line)


def read_mix_info(info_file):
  """Read a mix_info_file to form mix_info dictionary."""
  mix_info = {}
  with open(info_file, 'r') as f:
    for line in f:
      line = line.rstrip('\n')
      mix, rhs = line.split(' = ')
      operands = rhs.split(' + ')
      srcs = []
      rirs = []
      for operand in operands:
        src, rir = operand.split(' * ')
        srcs.append(src)
        rirs.append(rir)
      mix_info[mix] = [srcs, rirs]
  return mix_info


def write_item_dict(item_dict, item_file, separate=False):
  """Write tab separated source/rir lists in files for train/validate/eval."""
  if not separate:
    with open(item_file, 'w') as f:
      for subfolder in item_dict:
        for example in item_dict[subfolder]:
          line = '\t'.join([example] + item_dict[subfolder][example])
          f.write(line + '\n')
  else:
    for subfolder in item_dict:
      item_base, item_ext = item_file.split('.')
      item_file_sub = item_base + '_' + subfolder + '.' + item_ext
      with open(item_file_sub, 'w') as f:
        for example in item_dict[subfolder]:
          line = '\t'.join([example] + item_dict[subfolder][example])
          f.write(line + '\n')


def read_item_dict(item_file):
  """Reads an source/rir dict from a tab separated source/rir file."""
  item_dict = {}
  with open(item_file, 'r') as f:
    for line in f:
      line = line.rstrip('\n')
      fields = line.split('\t')
      item_key = fields[0]
      items = fields[1:]
      subfolder = items[0].split('/')[0]
      if subfolder not in item_dict:
        item_dict[subfolder] = {}
      item_dict[subfolder][item_key] = items
  return item_dict


def main():
  parser = argparse.ArgumentParser(
      description='Reverberates and mixes sources.\n The three folder names '
      'are always required. If a mix_info file is given, '
      'it is used to perform the mixing.\n Otherwise:\n'
      'If a list of sources is given to '
      'read from, it is used, otherwise the source folder is browsed to get a '
      'list of sources. Similarly for rirs, if a list is given it is used, '
      'otherwise the directory is browsed to get a list. Finally a mix_info '
      'file is made by randomly matching mixtures to rirs. After a '
      'mix_info file is made, the script needs to be run again by reading '
      'that generated mix_info file using the --read_mix_info argument '
      'along with the required folder arguments.')
  parser.add_argument(
      '-s', '--source_dir', help='Source directory.', required=True)
  parser.add_argument(
      '-r', '--rir_dir', help='RIR directory.', required=True)
  parser.add_argument(
      '-o', '--output_dir', help='Output directory.', required=True)
  parser.add_argument(
      '-p', '--part', help='Part number.', required=False, default=0,
      type=int)
  parser.add_argument(
      '-n', '--nparts', help='Number of parts to divide the mix_info list '
      'for parallel generation of reverberated mixtures from each sub-list.',
      required=False, default=1, type=int)
  parser.add_argument(
      '-m', '--num_mics', help='Number of mics.', required=False, default=1,
      type=int)
  parser.add_argument(
      '-sc', '--scale_rirs', help='Scale factor for RIRs.', required=False,
      default=10.0, type=float)
  parser.add_argument(
      '-w', '--write_mix_info',
      help='A file name to write out a list of to be reverberated and mixed '
      'sources and rirs.',
      default='/tmp/reverberate_and_mix_info_' + str(os.getpid()) + '.txt',
      required=False)
  parser.add_argument(
      '-e', '--read_mix_info',
      help='A file name to read a list of to be reverberated and mixed '
      'sources and rirs.',
      required=False)
  parser.add_argument(
      '-u', '--read_sources',
      help='A list file to read sources from. The format of the file is a tab '
      'separated list with a relative mixture path followed by relative '
      'source paths.', required=False)
  parser.add_argument(
      '-i', '--read_rirs',
      help='A list file to read rirs from. The format of the file is a tab '
      'separated list with an rir_key followed by relative rir paths.',
      required=False)
  parser.add_argument(
      '-ws', '--write_sources',
      help='Write a list file for sources. The format of the file is a tab '
      'separated list with a relative mixture path followed by relative '
      'source paths.', required=False)
  parser.add_argument(
      '-wr', '--write_rirs',
      help='Write a list file for rirs. The format of the file is a tab '
      'separated list with an rir_key followed by relative rir paths.',
      required=False)
  parser.add_argument(
      '-c', '--chat', type=bool, default=False, help='Enable chatty output.')
  parser.add_argument(
      '-oa', '--output_align', type=str, default='causal',
      choices=['causal', 'align_sources'],
      help='Alignment type of output signals. '
      'causal: Uses causal RIR filtering with no additional shift. '
      'align_sources: Find the average peak index of RIR(s) corresponding '
      '  each source and advance each source with that index. This has an '
      '  effect of aligning each source with their non-reverberated version.'
      'Note that using align_sources will make the average delay '
      '  between clean source signals and reverberated source signals zero.')
  parser.add_argument(
      '-rs', '--random_seed', help='Random seed.', required=False, default=123,
      type=int)
  args = parser.parse_args()

  if args.nparts > 1 and not args.read_mix_info:
    raise ValueError('When generating in parts, read from a '
                     'single mix_info file to avoid regenerating mix_info '
                     'randomly every time.')

  if args.read_mix_info:
    mix_info = read_mix_info(args.read_mix_info)
    print('Generating data using mix info in {}.'.format(args.read_mix_info))
    reverberate_and_mix(args.output_dir, args.source_dir, args.rir_dir,
                        mix_info, part=args.part, nparts=args.nparts,
                        num_mics=args.num_mics, scale_rirs=args.scale_rirs,
                        chat=args.chat, output_align=args.output_align)
  else:
    np.random.seed(args.random_seed)
    if args.read_sources:
      source_dict = read_item_dict(args.read_sources)
    else:
      source_dict = make_example_dict_from_folder(args.source_dir,
                                                  subfolder_events=None)
      if args.write_sources:
        write_item_dict(source_dict, args.write_sources)
        print('Tab separated sources written in {}'.format(args.write_sources))
    if args.read_rirs:
      rir_dict = read_item_dict(args.read_rirs)
    else:
      rir_dict = make_rir_dict_from_folder(args.rir_dir)
      if args.write_rirs:
        write_item_dict(rir_dict, args.write_rirs)
        print('Tab separated rirs written in {}'.format(args.write_rirs))
    mix_info = make_mix_info(source_dict, rir_dict)
  if args.write_mix_info and not args.read_mix_info:
    write_mix_info(mix_info, args.write_mix_info)
    print('Reverberate and mix info written in {}'.format(args.write_mix_info))
    print('Re-run this script with \"--read_mix_info {}\" argument along with '
          'the folder arguments to actually '
          'generate the data.'.format(args.write_mix_info))


if __name__ == '__main__':
  main()
