#!/bin/bash
ADCIRC=/home/atikr/asgs/opt/models/adcircs/adcirc-cg-v56.0.4.live.0-gfortran-wsl-ubuntu/work/adcirc
RUNDIR=/mnt/c/Users/atikr/OceanMesh2D/actual_amphan

cd "$RUNDIR" || { echo "ERROR: Cannot cd to $RUNDIR"; exit 1; }

echo "Cleaning old output files..."
rm -f fort.16 fort.33 fort.63 fort.64 fort.73 fort.74 Errfile PRINT run_stdout.log run_stderr.log

echo "Starting ADCIRC at $(date)"
"$ADCIRC" > run_stdout.log 2> run_stderr.log
EXIT_CODE=$?

echo "ADCIRC finished at $(date) with exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "--- STDERR ---"
    cat run_stderr.log
fi
