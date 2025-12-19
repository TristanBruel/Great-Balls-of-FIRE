#!/usr/bin/env bash
#
#SBATCH --job-name=linefinder
#SBATCH --output=linefinder.out
#SBATCH --error=linefinder.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=8:00:00

NCORES=6
JUGFILE='./m12i/linefinder_clustersofFIRE.py'

# Run JUG
source activate fire
for i in $(seq $NCORES); do jug execute $JUGFILE & done

# Workaround to check if JUG is done
while ! jug check $JUGFILE; do
   jug sleep-until $JUGFILE 
done


echo 
echo Done waiting!!!!
echo 
