#!/usr/bin/env bash
#
#SBATCH --job-name=BBHs_in_FIREbox
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3850
#SBATCH --time=6:00:00

startT=`date +%s`
start=$(date +"%T")
echo "Start time: $start"

source activate top
python -u master.py --sample Gamma --seed 2 > prints_bbhs_Gamma.txt
python -u master.py --sample ElBadry --seed 2 > prints_bbhs_EB.txt 
python -u master.py --sample SFR --seed 2 > prints_bbhs_SFR.txt

#python -u master_iso.py --sample SFR > prints_isobbhs_SFR.txt

endT=`date +%s`
end=$(date +"%T")
echo "End time: $end"

runtime=$((endT-startT))
echo "Runtime: $runtime s"
