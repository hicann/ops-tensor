#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -o pipefail

echo "ge_st_rt2: ${ge_st_rt2}"
echo "GIT_TARGET_BRANCH: ${GIT_TARGET_BRANCH}"
echo "ut_type: ${ut_type}"

########
# Init #
########

function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}

function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "[Command] ${date_time} ${cmd}"
   ${cmd}
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat ${log_path}
        fi
        echo "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}

REPOSITORY_NAME="ops-math"

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
rm -rf /home/jenkins/opensource/json
source /home/jenkins/Ascend/cann/bin/setenv.bash
main(){
    LOG_HEAD "Start run c++ testcase"
    LOG_DO sh build.sh --opkernel -u
    DP_ASSERT_EQUAL "$?" "0" "exec cmd: [sh build.sh -u --opkernel -u]"
}
main_param=$@
main $main_param
