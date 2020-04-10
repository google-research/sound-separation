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
r"""Makes DESED data lists.

Produces following output lists from DESED generated synthetic data:
${base_out}_train.txt
${base_out}_sources_train.txt
${base_out}_validation.txt
${base_out}_sources_validation.txt
${base_out}_eval.txt
${base_out}_sources_eval.txt
"""
import argparse
import glob
import re

VALIDATION_ITEMS = (
    '1875 01 3751 3127 2501 2503 1255 4379 1256 4380 3133 4386 1883 '
    '1884 1262 1886 1887 1888 4390 4393 3764 14 4396 645 646 3148 '
    '2522 647 1275 649 3772 1901 1279 23 3156 1281 1284 658 4410 '
    '3782 663 1919 1291 3166 2547 1926 4425 3176 3179 4429 1932 '
    '1933 3181 2557 3807 59 60 1944 4445 62 66 1319 3198 1950 70 '
    '3820 2571 72 3821 1325 1954 73 75 2574 3205 701 3208 80 4463 '
    '83 84 1964 3833 87 1341 1342 4474 3841 3842 95 1347 4477 3224 '
    '4478 3846 721 3228 2598 3849 724 1981 2603 3854 727 1983 3855 '
    '3236 109 1987 2611 4494 1361 113 3869 2621 1998 742 3253 1377 '
    '2633 2634 2009 2011 4515 2637 3883 2641 2642 4521 140 4526 1390 '
    '1392 2023 148 2653 3898 3278 2656 1400 4537 3904 3905 1404 2661 '
    '2662 3287 781 1408 3289 3912 1411 2668 166 1414 3921 4556 2676 '
    '794 1420 2677 3304 1424 803 180 3934 2686 3935 3311 2687 183 '
    '3313 184 2690 811 1438 2067 4571 1442 189 2070 191 4576 3949 '
    '3953 198 3328 4586 4587 1457 3336 2087 3964 3965 1464 833 1468 '
    '216 2720 3346 2097 218 839 4602 840 1477 844 3353 227 2734 2736 '
    '2737 233 3362 234 2115 3365 2116 2742 1491 859 3368 2120 4625 '
    '3372 1498 244 2123 246 3375 1502 2128 3379 875 4009 2759 4639 '
    '3384 3385 3386 4643 3388 2140 4647 263 889 1518 2770 3395 2145 '
    '4653 4024 2773 1523 1524 897 2150 4661 275 2782 1533 2784 4036 '
    '3409 3410 2785 3412 4039 1540 4043 2793 4678 918 2799 923 924 '
    '926 1554 4694 4695 298 300 4060 1561 3436 303 3438 941 2188 '
    '1570 2191 4713 3447 313 1576 950 1579 4083 3452 1580 2201 2828 '
    '4087 2204 322 3458 962 4729 1590 2211 967 4098 332 4735 3471 '
    '4739 1599 337 974 2220 3474 4742 4106 2225 2852 1607 4109 981 '
    '4110 4111 1610 3482 2856 351 2858 2231 1614 1615 2861 4118 '
    '4751 1618 356 991 1620 2865 359 994 3494 2870 363 2247 2248 '
    '374 4763 4765 1007 2884 2256 385 4775 1015 388 2889 2263 4151 '
    '395 1650 1651 2899 1026 1653 1655 4793 1034 4795 404 2284 3540 '
    '3542 2914 414 3554 4187 3558 2924 430 2933 432 2935 4202 4203 '
    '1704 1074 3583 4845 1084 4222 1715 2955 3594 2956 1716 4856 '
    '4857 457 459 3602 1725 2340 2966 2967 1099 4235 3608 466 468 '
    '2975 2350 471 1107 1108 474 477 479 2358 481 1116 4250 2989 '
    '485 2362 3624 1750 4888 4890 1753 2367 1755 1756 1757 3004 '
    '2375 2378 4277 3019 4911 3022 4289 3655 1148 4919 4293 3659 '
    '1153 4925 4300 3035 4303 3669 4313 4314 4315 1165 1796 4941 '
    '2413 4942 1169 4328 4952 3679 4954 3682 1812 4341 1182 4342 '
    '4344 3691 2427 552 2429 4352 1825 1192 4356 4358 2438 1206 569 '
    '3713 3082 3718 577 1216 2454 581 585 1858 2464 3095 2466 3097 '
    '3099 591 3108 1235 1236 3112 603 2482 606 3117 611 614 616')


def main():
  parser = argparse.ArgumentParser(
      description=('Make DESED train, validation and eval mixture and '
                   'sources lists.'))
  parser.add_argument(
      '-dta', '--desed_train_audio_dir', help='DESED train audio dir',
      required=True)
  parser.add_argument(
      '-dea', '--desed_eval_audio_dir', help='DESED eval audio dir',
      required=True)
  parser.add_argument(
      '-o', '--outbase', help='Base part of list files.', required=True)
  args = parser.parse_args()

  # No need to change below this line.
  trainval_mix_list = glob.glob('{dir}/*wav'.format(
      dir=args.desed_train_audio_dir))
  trainval_src_list = glob.glob('{dir}/*events/*wav'.format(
      dir=args.desed_train_audio_dir))
  trainval_mix_list = [w.replace(args.desed_train_audio_dir + '/', '')
                       for w in trainval_mix_list]
  trainval_src_list = [w.replace(args.desed_train_audio_dir + '/', '')
                       for w in trainval_src_list]

  validation_mix_list = VALIDATION_ITEMS.split(' ')

  print('There are {} validation items'.format(len(validation_mix_list)))

  # Train mix list is obtained by the difference between trainval
  # and validation mix lists.
  train_mix_list = []
  for mixture in trainval_mix_list:
    if mixture not in validation_mix_list:
      train_mix_list.append(mixture)

  # We also split sources between train and validation based on their mix
  # being in validation_mix_list or not.
  train_src_list = []
  validation_src_list = []
  for source in trainval_src_list:
    source_mix = re.match(r'(.*)_events.*', source).groups(0)[0]
    if source_mix in validation_mix_list:
      validation_src_list.append(source)
    else:
      train_src_list.append(source)

  # Write train and validation lists.
  with open('{}_train.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(train_mix_list) + '\n')
  with open('{}_validation.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(validation_mix_list) + '\n')
  with open('{}_sources_train.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(train_src_list) + '\n')
  with open('{}_sources_validation.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(validation_src_list) + '\n')

  # Obtain eval lists directly from the glob.glob.
  eval_mix_list = glob.glob('{dir}/*wav'.format(dir=args.desed_eval_audio_dir))
  eval_src_list = glob.glob('{dir}/*events/*wav'.format(
      dir=args.desed_eval_audio_dir))
  eval_mix_list = [w.replace(args.desed_eval_audio_dir + '/', '')
                   for w in eval_mix_list]
  eval_src_list = [w.replace(args.desed_eval_audio_dir + '/', '')
                   for w in eval_src_list]

  with open('{}_eval.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(eval_mix_list) + '\n')
  with open('{}_sources_eval.txt'.format(args.outbase), 'w') as f:
    f.write('\n'.join(eval_src_list) + '\n')

if __name__ == '__main__':
  main()
