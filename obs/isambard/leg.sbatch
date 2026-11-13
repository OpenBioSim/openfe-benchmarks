#!/bin/bash
#SBATCH --nodes=1
# Run one benchmark leg. Submitted by benchmarks.py with the run directory and the
# path to leg.py, since Slurm runs a copy of this script from its spool directory.

set -eo pipefail

RUN_DIR="$1"
LEG_PY="$2"

source "${RUN_DIR}/setup.sh"
set -u

python "${LEG_PY}" "${RUN_DIR}"
