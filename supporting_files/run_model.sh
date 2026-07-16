#!/bin/bash
set -e

ADCDIR=/home/atikr/asgs/opt/models/adcircs/adcirc-cg-v56.0.4.live.0-gfortran-wsl-ubuntu/work
export PATH=$PATH:$ADCDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/atikr/asgs/opt/lib

echo "Cleaning previous output..."
rm -rf fort.16 fort.33 fort.61.nc fort.62.nc fort.63.nc fort.64.nc fort.73.nc fort.74.nc swan_*.nc PE0000 PE0001

echo "Launching adcswan on a single core..."
adcswan

echo "SWAN+ADCIRC Run Completed!"
