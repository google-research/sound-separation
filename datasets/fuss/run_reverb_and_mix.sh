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

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 RAW_DATA_DIR BARE_GEN_DATA_DIR RANDOM_SEED"
  exit 1
fi

RAW_DATA_DIR=$1
BARE_GEN_DATA_DIR=$2
RANDOM_SEED=$3

# Define top level variables.
TOOLPATH=`dirname $0`
# This is the main directory where the prepared data will reside.
GEN_DATA_DIR=${BARE_GEN_DATA_DIR}_${RANDOM_SEED}

NPARTS_REVERB=24  # number of parallel jobs for reverb-and-mix.

# No need to change below this line

FSD_DIR=${RAW_DATA_DIR}/fsd_data
RIR_DIR=${RAW_DATA_DIR}/rir_data
MIX_DIR=${GEN_DATA_DIR}/ssdata
REVERB_MIX_DIR=${GEN_DATA_DIR}/ssdata_reverb

# Reverberate and remix data.

# Define variables for this part.
MIX_INFO=${REVERB_MIX_DIR}/mix_info.txt
SRC_LIST=${REVERB_MIX_DIR}/src_list.txt
RIR_LIST=${REVERB_MIX_DIR}/rir_list.txt
LOG_DIR=${REVERB_MIX_DIR}/log

if [ ! -d ${REVERB_MIX_DIR} ]; then
  mkdir -p ${REVERB_MIX_DIR}
else
  echo "${REVERB_MIX_DIR} exists. Please delete and re-run. Exiting."
  exit 1
fi

# Form src_list from train_example_list.txt validation_example_list.txt
# and eval_example_list.txt produced by make_ss_examples.py.

cp ${MIX_DIR}/*_example_list.txt ${REVERB_MIX_DIR}
cat ${REVERB_MIX_DIR}/*_example_list.txt > ${SRC_LIST}

mkdir -p ${LOG_DIR}

echo First make a mix_info file ${MIX_INFO}. Inspect the file to be sure.
echo If running again, this step can be skipped since the ${MIX_INFO} can be
echo used again.
python3 ${TOOLPATH}/reverberate_and_mix.py -s ${MIX_DIR} -r ${RIR_DIR} \
  -o ${REVERB_MIX_DIR} --write_mix_info ${MIX_INFO} --write_rirs ${RIR_LIST} \
  --read_sources ${SRC_LIST} --random_seed ${RANDOM_SEED}

echo Now, actually running generation with ${NPARTS_REVERB} parallel runs
echo in the background with produced mix_info file ${MIX_INFO}
for part in $(seq 0 $(( NPARTS_REVERB - 1 ))); do
python3 ${TOOLPATH}/reverberate_and_mix.py -s ${MIX_DIR} -r ${RIR_DIR} \
  -o ${REVERB_MIX_DIR} --read_mix_info ${MIX_INFO} \
  --part ${part} --nparts ${NPARTS_REVERB} --chat 1 > \
  ${LOG_DIR}/rev_and_mix_out_${part}_of_${NPARTS_REVERB} 2>&1 &
done
echo "Waiting for ${NPARTS_REVERB} background processes to finish! Check logs"
echo "in ${LOG_DIR}."
wait
echo Done!

# Let's check the reverb_mix folder by check_and_fix_folder.py
# There may be small mixture consistency errors due to writing sources in int16
# files whereas the mixture is calculated using floats originally.
python3 ${TOOLPATH}/check_and_fix_folder.py \
  -sd ${REVERB_MIX_DIR} -sl ${SRC_LIST}

end=$(date +"%T")
echo "Start time: $start, reverb and mix end time : $end"
