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
r"""A library of utilities."""
import os
import re
import numpy as np
import soundfile as sf


def read_wav(wav_f, always_2d=False):
  data, samplerate = sf.read(wav_f, always_2d=always_2d)
  return data, samplerate


# Use subtype = 'FLOAT' to write float wavs.
def write_wav(wav_f, wav_data, samplerate, subtype='PCM_16'):
  sf.write(wav_f, wav_data, samplerate, format='WAV', subtype=subtype)


def make_example_dict_from_folder(
    folder_sources, subset='all', ss_regex=re.compile(r'example.*_sources'),
    pattern='_sources', subfolder_events=('background', 'foreground')):
  """Returns a dictionary which maps subfolder -> example -> source wavs list.

  Returns a hierarchical dict of relative source file paths when given a
  folder produced by scaper.

  Args:
    folder_sources: Main path to a sources folder which contains
      train validation eval subfolders.
    subset: A subdirectory name or 'all' for all subdirectories.
    ss_regex: A regex that matches source folder names.
    pattern: The pattern that is assumed to be added after the base filename
      of the mixed file to get the source folder name.
    subfolder_events: Source/event subfolders under source folder, if any.
  Returns:
    A hierarchical dictionary as described above.
  """
  if subset == 'all':
    subfolders = ['train', 'validation', 'eval']
  else:
    subfolders = [subset]
  sources_for_mix = {}
  for subfolder in subfolders:
    sources_for_mix[subfolder] = {}
    src_sub = os.path.join(folder_sources, subfolder)
    if os.path.isdir(src_sub):
      src_entries = os.listdir(src_sub)
      src_selected = sorted(list(filter(ss_regex.search, src_entries)))
      for src_example in src_selected:
        src_example_base = src_example.rstrip(pattern)
        src_example_wav = src_example_base + '.wav'
        if not os.path.isfile(os.path.join(src_sub, src_example_wav)):
          raise ValueError('In {}, no mixed file {} but there is a folder '
                           'of sources {}'.format(
                               subfolder, src_example_wav, src_example))
        src_example_rel = os.path.join(subfolder, src_example_wav)
        sources_for_mix[subfolder][src_example_rel] = []
        if subfolder_events is not None:
          for ex_sub in subfolder_events:
            src_wav_dir = os.path.join(src_sub, src_example, ex_sub)
            if os.path.isdir(src_wav_dir):
              src_wavs = sorted(list(filter(re.compile(r'.*\.wav').search,
                                            os.listdir(src_wav_dir))))
              for src_wav in src_wavs:
                src_wav_f = os.path.join(src_wav_dir, src_wav)
                src_wav_f_rel = os.path.relpath(src_wav_f, folder_sources)
                sources_for_mix[subfolder][src_example_rel].append(
                    src_wav_f_rel)
        else:
          src_wav_dir = os.path.join(src_sub, src_example)
          if os.path.isdir(src_wav_dir):
            src_wavs = sorted(list(filter(re.compile(r'.*\.wav').search,
                                          os.listdir(src_wav_dir))))
            for src_wav in src_wavs:
              src_wav_f = os.path.join(src_wav_dir, src_wav)
              src_wav_f_rel = os.path.relpath(src_wav_f, folder_sources)
              sources_for_mix[subfolder][src_example_rel].append(src_wav_f_rel)
  return sources_for_mix


def make_example_list_from_folder(
    folder_sources, subset='all', ss_regex=re.compile(r'example.*_sources'),
    pattern='_sources', subfolder_events=('background', 'foreground')):
  """Makes a tab separated list of examples from a top folder."""
  example_dict = make_example_dict_from_folder(
      folder_sources, subset=subset, ss_regex=ss_regex, pattern=pattern,
      subfolder_events=subfolder_events)
  example_list = []
  for subset in example_dict:
    for example in example_dict[subset]:
      example_list.append('\t'.join([example] + example_dict[subset][example]))
  return example_list


def check_and_correct_example(example, root_dir,
                              check_length, fix_length,
                              check_mix, fix_mix,
                              sample_rate=16000, duration=10.0,
                              chat=False):
  """Checks and possibly corrects a scaper produced example."""
  # Earlier versions of scaper had a tendency to make mistakes every
  # once in a while.
  # This has most likely been fixed in the latest scaper release, at least
  # for the parameter settings we are using, but this test and correction
  # can serve to catch failures that may be introduced by using the wrong
  # scaper version, or by using parameters in scaper that do not maintain
  # mixture consistency.  For example, at the time of this coding,
  # using scaper reverb breaks mixture consistency.

  # Enforce dependencies between flags.
  if fix_mix:
    check_mix = True
  if check_mix:
    fix_length = True
  if fix_length:
    check_length = True

  length_problem = 0
  fixed_length = 0
  mix_problem = 0
  fixed_mix = 0

  files = example.split('\t')
  mixfile = files[0]
  if chat:
    print('Checking {}'.format(mixfile))
  components = files[1:]

  def resize_audio(audio, length):
    in_length = audio.shape[0]
    new_audio = np.zeros((length, audio.shape[1]), dtype=audio.dtype)
    new_audio[0:min(length, in_length), :] = \
        audio[0:min(length, in_length), :]
    return new_audio

  expected_samples = int(duration * sample_rate)

  if check_length:

    for myfile in files:
      file_abs = os.path.join(root_dir, myfile)
      file_info = sf.info(file_abs)
      num_samples = int(file_info.duration * file_info.samplerate)

      if num_samples != expected_samples:
        length_problem += 1
        print('Warning: scaper output on {:s} is {:d} samples; '
              'expected {:d}'.format(file_abs, num_samples, expected_samples))

        audio, _ = read_wav(file_abs, always_2d=True)
        num_samples, num_channels = audio.shape
        audio = resize_audio(audio, expected_samples)

        if fix_length:
          # rewrite corrected source
          print('Adjusting length of {:s}'.format(file_abs))
          write_wav(file_abs, audio, sample_rate, subtype=file_info.subtype)
          fixed_length += 1

  def check_mixture(mixed_data, remixed_data, mixfile):
    if not np.allclose(mixed_data, remixed_data, rtol=1e-4, atol=1e-5):
      mixed_norm = np.linalg.norm(mixed_data)
      err_norm = np.linalg.norm(mixed_data - remixed_data)
      normalized_err = err_norm / mixed_norm
      print('WARNING: Mismatched mixed data found {}. '
            'Normalized error {}, mixed signal norm {}'.format(
                mixfile, normalized_err, mixed_norm))
      return False
    return True

  if check_mix:
    mixfile_abs = os.path.join(root_dir, mixfile)
    mix_info = sf.info(mixfile_abs)
    mixture, _ = read_wav(mixfile_abs, always_2d=True)
    num_samples, num_channels = mixture.shape

    source_sum = np.zeros((expected_samples, num_channels),
                          dtype=mixture.dtype)
    for srcfile in components:
      srcfile_abs = os.path.join(root_dir, srcfile)
      source, _ = read_wav(srcfile_abs, always_2d=True)
      # sum up sources
      source_sum += source

    if not check_mixture(mixture, source_sum, mixfile):
      mix_problem += 1
      if fix_mix:
        print('Rewriting corrected mixture file {}.'.format(mixfile_abs))
        # write new mixture
        # note we are not doing anything about clipping here,
        # so if mismatch is due to clipping it will clip again on writing.
        mixture = source_sum
        write_wav(mixfile_abs, mixture, sample_rate, subtype=mix_info.subtype)
        fixed_mix += 1
  return length_problem, fixed_length, mix_problem, fixed_mix
