#!/bin/bash
#
# SLURM script to run Edison query for materials
#
# Usage:
#   sbatch run_edison_query.slurm OUTPUT_DIR

#SBATCH --job-name=edison
#SBATCH --output=logs/edison_%j.out
#SBATCH --error=logs/edison_%j.err
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64GB                           # server memory (MBs) requested (per node)
#SBATCH --time=1-00:00:00                    # maximum walltime (HH:MM:SS)
#SBATCH --partition=kilian                    # partition (queue)

OUTPUT_DIR=$1
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR is not set"
    exit 1
fi

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Activate conda environment
# echo "Activating conda environment: sci-llm"
# conda activate sci-llm

# Verify environment
# echo "Python: $(which python)"
# echo "Python version: $(python --version)"
# echo ""

# Check for EDISON_API_KEY
if [ -z "$EDISON_API_KEY" ]; then
    echo "ERROR: EDISON_API_KEY environment variable is not set"
    echo "Please set it with: conda env config vars set EDISON_API_KEY=your_key"
    exit 1
fi

# Run the Edison query script
echo "Starting Edison query..."
echo "Command: python query_materials_with_edison.py -od $OUTPUT_DIR"
echo ""

python query_materials_with_edison.py -bs 10 -od $OUTPUT_DIR --input data/cleaned_stoic_filtered_downfolded_dataset.csv --verbose

# Check exit status
EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"

exit $EXIT_CODE

