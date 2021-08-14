#!/bin/bash -l
#$ -l h_rt=23:59:59
#$ -N DQN-Job-BigV2-Job
#$ -o DQN_logs/BigV2-output
#$ -e DQN_logs/SmallV2-errors
#$ -m beas
#$ -P cs542sp
#$ -l gpus=1
#$ -l gpu_c=6.0

# Combine output and error files into a single file
# Specify the output file name

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="


module load python3/3.8.10
module load tensorflow/2.5.0
source /projectnb/cs542sp/nikzad/Famished-Geese/geese-tf-env/bin/activate
python3 train.py 