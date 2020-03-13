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

SCRIPT_PATH=`dirname $0`

source ${SCRIPT_PATH}/setup.sh

# No need to change below this line

SS_DIR=${DEV_DATA_DIR}/ssdata
SS_REVERB_DIR=${DEV_DATA_DIR}/ssdata_reverb

# Download and unarchive SS data.
# ssdata.tar.gz should have a directory ssdata which has
# train, validation, eval subdirectories underneath
# and under those there should be a example wav files that will be used
# as train, validation and eval mixtures.
mkdir -p ${DOWNLOAD_DIR}
mkdir -p ${DEV_DATA_DIR}
if [ ! -s ${DOWNLOAD_DIR}/ssdata.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/ssdata.tar.gz ${SSDATA_URL}
else
  echo "${DOWNLOAD_DIR}/ssdata.tar.gz exists, skipping download."
fi

if [ ! -d ${SS_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/ssdata.tar.gz -C ${DEV_DATA_DIR}
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

if [ ! -d ${SS_REVERB_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz -C ${DEV_DATA_DIR}
else
  echo "${SS_REVERB_DIR} directory exists, skipping unarchiving."
fi

unarchive_time=$(date +"%T")
echo "Start time : $start, download and unarchive finish time: $unarchive_time"
