#!/bin/bash
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

start=$(date +"%T")
echo "Start time : $start"

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 RAW_DATA_DIR BARE_GEN_DATA_DIR RANDOM_SEED NUM_TRAIN NUM_VAL"
  exit 1
fi

RAW_DATA_DIR=$1
BARE_GEN_DATA_DIR=$2
RANDOM_SEED=$3
NUM_TRAIN_MIX=$4
NUM_VALIDATION_MIX=$5
NUM_EVAL_MIX=0

# Actual GEN_DATA_DIR to use for data generation outputs.
GEN_DATA_DIR=${BARE_GEN_DATA_DIR}_${RANDOM_SEED}

# Define top level variables.
TOOLPATH=`dirname $0`


# No need to change below this line

FSD_DIR=${RAW_DATA_DIR}/fsd_data
MIX_DIR=${GEN_DATA_DIR}/ssdata
REVERB_MIX_DIR=${GEN_DATA_DIR}/ssdata_reverb

# Makes foreground and background wav file lists from fsd_data used as
# foreground and background events in scaper.
python3 ${TOOLPATH}/make_fg_bg_file_lists.py --data_dir ${FSD_DIR}

# Runs scaper to obtain the desired amount of example mixed signals.
# It also saves the individual source wavs used in mixture wavs.
python3 ${TOOLPATH}/make_ss_examples.py -f ${FSD_DIR} -b ${FSD_DIR} \
  -o ${MIX_DIR} --allow_same 1 --num_train ${NUM_TRAIN_MIX} \
  --num_validation ${NUM_VALIDATION_MIX} --num_eval ${NUM_EVAL_MIX} \
  --random_seed ${RANDOM_SEED}

scaper_time=$(date +"%T")
echo "Start time : $start, scaper end time: $scaper_time"

# Optionally we can run check_and_fix_folder.py to check output folder.
# This is disabled since the check is done within make_ss_examples.py now.

# python3 ${TOOLPATH}/check_and_fix_folder.py -sd ${MIX_DIR}

