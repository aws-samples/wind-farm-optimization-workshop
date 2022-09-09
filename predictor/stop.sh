#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Kill the primary component...
PID="`ps -ef | grep wind_farm_predictor.py | grep -v grep | awk '{print $2}'`"
if [ ! -z "${PID}" ]; then
   echo "Killing wind_farm_predictor.py PID:" ${PID} "..."
   kill -9 ${PID}
else
   echo "INFO: wind_farm_predictor.py does not appear to be running"
fi

# Exit
exit 0
