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

FUSS_DIR=${FUSS_ROOT_DIR}/ssdata/
DESED_LIST_DIR=${DESED_ROOT_DIR}/DESED_synthetic_2020_lists
DESED_TRAIN_AUDIO_DIR=${DESED_ROOT_DIR}/dataset/audio/train/synthetic20/soundscapes
DESED_EVAL_AUDIO_DIR=${DESED_ROOT_DIR}/dataset/audio/eval/synthetic20/soundscapes/fbsnr_30dB
OUTBASE=DESED_synthetic_2020_audio

mkdir -p ${DESED_LIST_DIR}
mkdir -p ${MODEL_DIR}

bash ${SCRIPT_PATH}/generate_desed_eval_data.sh

python3 ${SCRIPT_PATH}/make_desed_lists.py \
 --desed_train_audio_dir ${DESED_TRAIN_AUDIO_DIR} \
 --desed_eval_audio_dir ${DESED_EVAL_AUDIO_DIR} \
 --outbase ${DESED_LIST_DIR}/${OUTBASE}

for subset in "train" "validation" "eval"; do
  python3 ${SCRIPT_PATH}/convert_desed_lists.py \
    --mixtures ${DESED_LIST_DIR}/${OUTBASE}_${subset}.txt \
    --sources ${DESED_LIST_DIR}/${OUTBASE}_sources_${subset}.txt \
    --output ${MODEL_DIR}/${OUTBASE}_sslist_${subset}.txt
done

python3 ${SCRIPT_PATH}/make_mixing_list.py \
  --out ${MODEL_DIR}/FUSS_DESED_2_train_mixture_bg_fg_list.txt \
  --dirs ${FUSS_DIR} ${DESED_TRAIN_AUDIO_DIR} \
  --lists ${FUSS_DIR}/train_example_list.txt ${MODEL_DIR}/${OUTBASE}_sslist_train.txt \
  --modes 'mixture' 'sources' --data_names FUSS DESED \
  --num 20000 --random_seed 2020 \
  --class_maps 'DESED_bg_sins,BG_DSD;DESED_[^:]*,FG_DSD'

python3 ${SCRIPT_PATH}/make_mixing_list.py \
  --out ${MODEL_DIR}/FUSS_DESED_2_validation_mixture_bg_fg_list.txt \
  --dirs ${FUSS_DIR} ${DESED_TRAIN_AUDIO_DIR} \
  --lists ${FUSS_DIR}/validation_example_list.txt ${MODEL_DIR}/${OUTBASE}_sslist_validation.txt \
  --modes 'mixture' 'sources' --data_names FUSS DESED \
  --num 1000 --random_seed 2020 \
  --class_maps 'DESED_bg_sins,BG_DSD;DESED_[^:]*,FG_DSD'

python3 ${SCRIPT_PATH}/make_mixing_list.py \
  --out ${MODEL_DIR}/FUSS_DESED_2_eval_mixture_bg_fg_list.txt \
  --dirs ${FUSS_DIR} ${DESED_EVAL_AUDIO_DIR} \
  --lists ${FUSS_DIR}/eval_example_list.txt ${MODEL_DIR}/${OUTBASE}_sslist_eval.txt \
  --modes 'mixture' 'sources' --data_names FUSS DESED \
  --num 1000 --random_seed 2020 \
  --class_maps 'DESED_bg_sins,BG_DSD;DESED_[^:]*,FG_DSD'
