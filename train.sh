#!/bin/sh
#
# Submit using:  sbatch <script-file>
# Status check:  squeue
#
#*******************************************************************************

# SLURM options

# Job name

#SBATCH -J mesa

# Run directory

#SBATCH -D /storage2/sl84/

# Run on partition

#SBATCH -p normal

# Resource limits
#   ntasks            max number of tasks that will be used
#   ntasks-per-socket max number of tasks per socket to use
#
#   Specify one or the other of these.
#   
#   mem               memory per node in MB
#   mem-per-cpu       memory per core in MB
#
#   time:             maximum wall clock time (hh:mm:ss)

#SBATCH --ntasks=32
#SBATCH --time=72:00:00


# Output

# Mail (optional)
#   send mail when the job ends to addresses specified by --mail-user

#SBATCH --mail-type=end
#SBATCH --mail-user=sl84@illinois.edu

# Export environment variables to job

#SBATCH --export=ALL

#*******************************************************************************

# Script commands

export OMP_NUM_THREADS=$SLURM_TASKS_PER_NODE

echo Running with $np threads at `date`

python train_UNET.py

echo job complete at `date`
#*******************************************************************************

# Done.
