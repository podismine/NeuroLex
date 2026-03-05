#!/bin/bash
#SBATCH -J fmriprep
#SBATCH --partition=all
#SBATCH --array=0-999
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/fmriprep_%A_%a.out
#SBATCH -e logs/fmriprep_%A_%a.err
module load freesurfer
module load singularity

set -euo pipefail
offset=0

BIDS_DIR=/path/to/bids/
OUT_DIR=/path/to/fmriprep
WORK_DIR=/path/to/work/
IMG=/path/to/fmriprep-stable.simg 
SUBJECTS=/path/to/subjects.txt

mkdir -p logs "$WORK_DIR" "$OUT_DIR"

line=$((SLURM_ARRAY_TASK_ID + offset))
SUBJECT=$(sed -n "${line}p" "$SUBJECTS")

echo "[$(date)] Running fMRIPrep for $SUBJECT"

singularity run \
  -B $BIDS_DIR:/data \
  -B $OUT_DIR:/out \
  -B $WORK_DIR:/work \
  $IMG \
  /data /out participant \
  --participant-label ${SUBJECT#sub-} \
  --fs-license-file /path/to/license.txt \
  --work-dir /work/${SUBJECT} \
  --nthreads $SLURM_CPUS_PER_TASK \
  --omp-nthreads 1 \
  --mem_mb 32000 \
  --skip-bids-validation \
  --clean-workdir \
  --output-spaces MNI152NLin6Asym:res-2 T1w \
  --stop-on-first-crash \
  --fs-no-reconall
