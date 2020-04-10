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

# Download and unarchive model data.
# The FUSS_baseline_dry_model.tar.gz file should have 3 files that constitute
# the baseline checkpoint:
# baseline_dry_inference.meta
# baseline_dry_model.data-00000-of-00001
# baseline_dry_model.index
mkdir -p ${DOWNLOAD_DIR}
mkdir -p ${MODEL_DIR}
if [ ! -s ${DOWNLOAD_DIR}/FUSS_baseline_dry_model.tar.gz ]; then
  curl --output ${DOWNLOAD_DIR}/FUSS_baseline_dry_model.tar.gz ${BASELINE_DRY_MODEL_URL}
else
  echo "${DOWNLOAD_DIR}/FUSS_baseline_dry_model.tar.gz exists, skipping download."
fi

if [ ! -d ${BASELINE_DRY_MODEL_DIR} ]; then
  tar xzf ${DOWNLOAD_DIR}/FUSS_baseline_dry_model.tar.gz -C ${MODEL_DIR}
else
  echo "${BASELINE_DRY_MODEL_DIR} directory exists, skipping unarchiving."
fi
