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

# Makes hybrid file lists for combining DESED and FUSS data for training
# separation models with both in-domain (DESED) data and arbitrary domain (FUSS)
# data

SCRIPT_PATH=`dirname $0`

source ${SCRIPT_PATH}/setup.sh

DESED_CODE_PATH=${DESED_ROOT_DIR}/synthetic/code
DESED_EVAL_AUDIO_DIR=${DESED_ROOT_DIR}/dataset/audio/eval/synthetic20/soundscapes

pushd ${DESED_CODE_PATH}

for db in 30dB; do
  if [ ! -d ${DESED_ROOT_DIR}/synthetic/audio/eval/soundscapes_generated_fbsnr/${db} ]; then
    python3 generate_eval_FBSNR.py
  else
    echo "Not generating jams folder ${DESED_ROOT_DIR}/synthetic/audio/eval/soundscapes_generated_fbsnr/${db} since it exists!"
    echo "To regenerate jams, delete this folder."
  fi
  if [ ! -d ${DESED_EVAL_AUDIO_DIR}/fbsnr_${db} ]; then
    python3 generate_wav.py \
      --jams_folder=${DESED_ROOT_DIR}/synthetic/audio/eval/soundscapes_generated_fbsnr/${db} \
      --soundbank=${DESED_ROOT_DIR}/synthetic/audio/eval/soundbank \
      --out_audio_dir=${DESED_EVAL_AUDIO_DIR}/fbsnr_${db} \
      --out_tsv ${DESED_ROOT_DIR}/dataset/metadata/eval/synthetic20/eval_fbsnr_${db}.csv \
      --save_isolated
  else
    echo "Not generating wavs folder ${DESED_EVAL_AUDIO_DIR}/fbsnr_${db} since it exists!"
    echo "To regenerate wavs, delete this folder."
  fi
done

popd +0
