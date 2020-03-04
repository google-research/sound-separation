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

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DOWNLOAD_DIR RAW_DIR"
  exit 1
fi

DOWNLOAD_DIR=$1
RAW_DIR=$2

# The fsd data archive file is assumed to include a top folder named fsd_data.
FSD_DATA_URL="https://zenodo.org/record/3694384/files/FUSS_fsd_data.tar.gz"
# The rir data archive file is assumed to include a top folder named rir_data.
RIR_DATA_URL="https://zenodo.org/record/3694384/files/FUSS_rir_data.tar.gz"

# No need to change below this line

FSD_DIR=${RAW_DIR}/fsd_data
RIR_DIR=${RAW_DIR}/rir_data

# Download and unarchive FSD data.
# fsd_data.tar.gz should have a directory fsd_data which has
# train, validation, eval subdirectories underneath
# and under those there should be a single folder called sound.
# under the sound folder there should be multiple wav files that will be used
# in train, validation and eval mixtures.
mkdir -p ${DOWNLOAD_DIR}
mkdir -p ${RAW_DIR}

if [ ! -s ${DOWNLOAD_DIR}/fsd_data.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/fsd_data.tar.gz ${FSD_DATA_URL}
else
  echo "${DOWNLOAD_DIR}/fsd_data.tar.gz exists, skipping download."
fi

if [ ! -d ${FSD_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/fsd_data.tar.gz -C ${RAW_DIR}
else
  echo "${FSD_DIR} directory exists, skipping unarchiving."
fi

# Download and unarchive RIR data.
# rir_data.tar.gz should have a top level directory called rir_data and then
# train, validation and eval subdirectories and
# under those there will be rir_* folders each one containing foreground and
# background rir wav files named as foreground0.wav and background0.wav etc.
# The RIRs can be used with foreground and background components of each
# scaper mixture.
if [ ! -s ${DOWNLOAD_DIR}/rir_data.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/rir_data.tar.gz ${RIR_DATA_URL}
else
  echo "${DOWNLOAD_DIR}/rir_data.tar.gz exists, skipping download."
fi

if [ ! -d ${RIR_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/rir_data.tar.gz -C ${RAW_DIR}
else
  echo "${RIR_DIR} directory exists, skipping unarchiving."
fi

unarchive_time=$(date +"%T")
echo "Start time : $start, download and unarchive finish time: $unarchive_time"
