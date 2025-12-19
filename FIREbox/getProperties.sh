#!/usr/bin/env bash
#
#SBATCH --job-name=FIREbox_galaxies
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3850
#SBATCH --time=24:00:00

startT=`date +%s`
start=$(date +"%T")
echo "Start time: $start"

module load anaconda3
source activate top
python -u haloProperties.py > prints_halos.txt 
python -u SFR_Rvir.py > prints_SFR_Rvir.txt

endT=`date +%s`
end=$(date +"%T")
echo "End time: $end"

runtime=$((endT-startT))
echo "Runtime: $runtime s"
