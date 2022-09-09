#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

LOCATION="$1"
LOGFILE="/tmp/optimizer.log"

if [ -f ${LOGFILE} ]; then
    mv ${LOGFILE} ${LOGFILE}-$$
fi

env > ${LOGFILE}
while true
do
    python3 -u ${LOCATION}/wind_farm_optimizer.py 1>> ${LOGFILE} 2>&1
    sleep 10
done
