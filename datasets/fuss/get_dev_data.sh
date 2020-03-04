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

# Download directory to download data into.
DOWNLOAD_DIR=/data/download

# This is the main directory where the fixed dev data will reside.
ROOT_DIR=/data/dcase2020_task4_ss_dev

# The ssdata archive file is assumed to include a top folder named ssdata.
SSDATA_URL="https://zenodo.org/record/3694384/files/FUSS_ssdata.tar.gz"
# The archive file is assumed to include a top folder named ssdata_reverb.
SSDATA_REVERB_URL="https://zenodo.org/record/3694384/files/FUSS_ssdata_reverb.tar.gz"

# No need to change below this line

SS_DIR=${ROOT_DIR}/ssdata
SS_REVERB_DIR=${ROOT_DIR}/ssdata_reverb

# Download and unarchive SS data.
# ssdata.tar.gz should have a directory ssdata which has
# train, validation, eval subdirectories underneath
# and under those there should be a example wav files that will be used
# as train, validation and eval mixtures.
mkdir -p ${DOWNLOAD_DIR}
mkdir -p ${ROOT_DIR}
if [ ! -s ${DOWNLOAD_DIR}/ssdata.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/ssdata.tar.gz ${SSDATA_URL}
else
  echo "${DOWNLOAD_DIR}/ssdata.tar.gz exists, skipping download."
fi

if [ ! -d ${SS_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/ssdata.tar.gz -C ${ROOT_DIR}
else
  echo "${SS_DIR} directory exists, skipping unarchiving."
fi

# Download and unarchive SSREVERB data.
# ssdata_reverb.tar.gz should have a top level directory called ssdata_reverb
# and then train, validation and eval subdirectories and
# under those the same structure as ssdata.
if [ ! -s ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz ${SSDATA_REVERB_URL}
else
  echo "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz exists, skipping download."
fi

if [ ! -d ${SS_REVERB_DIR ]; then
  tar xzf ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz -C ${ROOT_DIR}
else
  echo "${SS_REVERB_DIR} directory exists, skipping unarchiving."
fi

unarchive_time=$(date +"%T")
echo "Start time : $start, download and unarchive finish time: $unarchive_time"
