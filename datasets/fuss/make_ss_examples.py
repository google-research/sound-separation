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
r"""A script to mix sources.

Usage example:
python3 make_ss_examples.py -f ${FSD_DIR} -b ${FSD_DIR} \
  -o ${MIX_DIR} --allow_same 1 --num_train ${NUM_TRAIN_MIX} \
  --num_validation ${NUM_VALIDATION_MIX} --num_eval ${NUM_EVAL_MIX} \
  --random_seed ${RANDOM_SEED}
"""

import argparse
import os

import numpy as np
import scaper

from utils import check_and_correct_example


class Mixer(object):
  """A class that calls scaper to mix sources."""

  def __init__(self, output_root, fg_root, bg_root, allow_same_label=False,
               num_train=200, num_validation=50, num_eval=50, random_seed=123):
    self.fg_root = fg_root
    self.bg_root = bg_root
    self.output_root = output_root
    if os.path.isdir(self.output_root):
      raise ValueError('Output root folder {} already exists. Please '
                       'delete and re-run.'.format(self.output_root))
    self.allow_same_label = allow_same_label
    self.filestem = 'example'

    # Scaper settings

    # random seed for this Scaper object
    self.random_seed = random_seed
    np.random.seed(random_seed)

    # global settings
    self.sample_rate = 16000

    self.num_examples = {'train': num_train, 'eval': num_eval,
                         'validation': num_validation}

    self.ref_db = -55
    self.duration = 10.0
    self.snr_spec = ('uniform', -5, 25)

    self.min_fg_events = 0
    self.max_fg_events = 3

    self.reverb = None  # we are providing reverb outside of scaper
    # warning: Do not enable reverb in scaper:
    # it will break mixture consistency of the output
    self.save_sources = True  # we need reference sources for separation

    self.disable_sox_warnings = True
    self.disable_instantiation_warnings = True
    self.no_audio = False

    # event distributions

    # longer than the longest source file
    self.source_time_spec = ('uniform', 0.0, 1000.0)

    self.event_time_spec = ('uniform', 0.0, self.duration)
    self.event_duration_spec = ('const', self.duration)

    self.pitch_spec = None
    self.time_stretch_spec = None

    # flags for error checking and correction
    self.check_scaper_length_errors = True
    self.fix_scaper_length_errors = True    # implies check_scaper_length_errors
    self.check_scaper_mixture_errors = True  # implies all of the above
    self.fix_scaper_mixture_errors = True   # implies all of the above

  def get_file_list(self, subset, style):
    """Get file list with relative paths to the desired subset."""
    list_name = os.path.join(self.fg_root, subset + '_' + style + '.txt')

    with open(list_name, 'r') as f:
      file_list = f.read().splitlines()

    # Pick the relative path wrt subset name.
    # This is required since scaper checks the parent folder of the wav files
    # and the parent folder name should be from the list of allowed labels.
    file_list = [os.path.relpath(f, subset) for f in file_list]
    np.random.shuffle(file_list)
    return file_list

  def mix_all_subsets(self):
    all_examples = []
    for subset in ['eval', 'validation', 'train']:
      if subset in self.num_examples:
        if self.num_examples[subset] > 0:
          example_list = self.mix_subset(subset)
          output_list_file = subset + '_example_list.txt'
          self.write_list_file(example_list, output_list_file)
          all_examples.extend(example_list)
    return all_examples

  def mix_subset(self, subset):
    """Mixes examples in a subset folder."""
    num_examps = self.num_examples[subset]

    print('Preparing {} examples.'.format(subset))

    os.makedirs(os.path.join(self.output_root, subset), exist_ok=False)

    fg_file_list = self.get_file_list(subset, 'foreground')
    bg_file_list = self.get_file_list(subset, 'background')

    ind_fg = 0
    ind_bg = 0

    example_list = []
    fg_folder = os.path.join(self.fg_root, subset)
    bg_folder = os.path.join(self.bg_root, subset)

    # Create a scaper object, folders are used to get labels.
    self.sc = scaper.Scaper(self.duration, fg_folder,
                            bg_folder, random_state=self.random_seed)
    self.sc.protected_labels = []
    self.sc.ref_db = self.ref_db
    self.sc.sr = self.sample_rate

    for n in range(num_examps):
      print('Generating example: {:d}/{:d}'.format(n+1, num_examps))
      example, ind_fg, ind_bg = self.generate_example(
          subset, n, num_examps, bg_file_list, fg_file_list,
          ind_fg, ind_bg)
      example_list.append(example)
    return example_list

  def add_fg_event(self, source_file, label):
    # add a single event
    self.sc.add_event(label=('const', label),
                      source_file=('const', source_file),
                      source_time=self.source_time_spec,
                      event_time=self.event_time_spec,
                      event_duration=self.event_duration_spec,
                      snr=self.snr_spec,
                      pitch_shift=self.pitch_spec,
                      time_stretch=self.time_stretch_spec)

  def add_bg_event(self, source_file, label):
    self.sc.add_background(label=('const', label),
                           source_file=('const', source_file),
                           source_time=self.source_time_spec)

  def generate_example(self, subset, n, num_examples,
                       bg_file_list, fg_file_list, ind_fg, ind_bg):
    """Generates a mixed example."""
    # reset the event specifications for foreground and background at the
    # beginning of each loop to clear all previously added events
    self.sc.bg_spec = []
    self.sc.fg_spec = []
    used_labels = []

    def add_events(num_events, file_list, ind_file, used_labels, add_event_fn,
                   file_folder):
      """Add foreground or background event."""
      # add a number of events
      num_files = len(file_list)

      for _ in range(num_events):

        label = None

        while label is None or label in used_labels:

          source_file = file_list[ind_file]
          label = source_file.split('/')[0]

          ind_file = (ind_file + 1) % num_files

          # shuffle file list when it turns over to zero
          if ind_file == 0:
            np.random.shuffle(file_list)
          if self.allow_same_label:
            break

        add_event_fn(os.path.join(file_folder, source_file), label)
        used_labels.append(label)

      return ind_file, used_labels

    # Add background event, making sure that labels do not repeat except when
    # self.allow_same_label is True.
    num_bg_events = 1
    ind_bg, used_labels = add_events(num_bg_events, bg_file_list,
                                     ind_bg, used_labels, self.add_bg_event,
                                     os.path.join(self.bg_root, subset))

    # Add some number of foreground events.
    num_fg_events = np.random.randint(self.min_fg_events,
                                      self.max_fg_events + 1)
    ind_fg, used_labels = add_events(num_fg_events,
                                     fg_file_list,
                                     ind_fg,
                                     used_labels,
                                     self.add_fg_event,
                                     os.path.join(self.fg_root, subset))

    # generate
    num_digits = len(str(num_examples))
    fmt = '{0:0' + str(num_digits) + 'd}'
    n_str = fmt.format(n)
    mixfile = os.path.join(subset, self.filestem + n_str + '.wav')
    jamsfile = os.path.join(subset, self.filestem + n_str + '.jams')
    txtfile = os.path.join(subset, self.filestem + n_str + '.txt')
    src_dir = os.path.join(subset, self.filestem + n_str + '_sources')

    self.sc.generate(os.path.join(self.output_root, mixfile),
                     os.path.join(self.output_root, jamsfile),
                     allow_repeated_label=True,
                     allow_repeated_source=True,
                     save_isolated_events=self.save_sources,
                     isolated_events_path=os.path.join(self.output_root, src_dir),
                     reverb=self.reverb,
                     disable_sox_warnings=self.disable_sox_warnings,
                     disable_instantiation_warnings= \
                     self.disable_instantiation_warnings,
                     no_audio=self.no_audio,
                     txt_path=os.path.join(self.output_root, txtfile))

    bg_file = os.path.join(src_dir, 'background0_sound.wav')
    fg_files = [os.path.join(
        src_dir, 'foreground{:d}_sound.wav'.format(i))
                for i in range(num_fg_events)]
    files = [mixfile, bg_file] + fg_files
    example = '\t'.join(files)
    return example, ind_fg, ind_bg

  def check_and_correct_list_of_examples(self, example_list):
    for example in example_list:
      check_and_correct_example(
          example, self.output_root,
          self.check_scaper_length_errors, self.fix_scaper_length_errors,
          self.check_scaper_mixture_errors, self.fix_scaper_mixture_errors,
          sample_rate=self.sample_rate,
          duration=self.duration)

  def write_list_file(self, example_list, output_list_file):
    with open(os.path.join(self.output_root, output_list_file), 'w') as f:
      f.write('\n'.join(example_list) + '\n')


def main():
  parser = argparse.ArgumentParser(
      description='Mixes sources to produce mixed files.')
  parser.add_argument(
      '-f', '--fg_dir', help='Foreground sources directory.', required=True)
  parser.add_argument(
      '-b', '--bg_dir', help='Background sources directory.', required=True)
  parser.add_argument(
      '-o', '--output_dir', help='Output directory.', required=True)
  parser.add_argument(
      '-a', '--allow_same', type=bool, default=False,
      help='Allow same label in a mixture, this is necessary when using a '
      'single label class.')
  parser.add_argument(
      '-nt', '--num_train', type=int, default=200,
      help='Number of training examples to generate.')
  parser.add_argument(
      '-nv', '--num_validation', type=int, default=50,
      help='Number of validation examples to generate.')
  parser.add_argument(
      '-ne', '--num_eval', type=int, default=50,
      help='Number of eval examples to generate.')
  parser.add_argument(
      '-rs', '--random_seed', help='Random seed.', required=False, default=123,
      type=int)
  args = parser.parse_args()

  mixer = Mixer(args.output_dir, args.fg_dir, args.bg_dir,
                allow_same_label=args.allow_same,
                num_train=args.num_train,
                num_validation=args.num_validation,
                num_eval=args.num_eval,
                random_seed=args.random_seed)
  example_list = mixer.mix_all_subsets()
  mixer.check_and_correct_list_of_examples(example_list)

if __name__ == '__main__':
  main()
