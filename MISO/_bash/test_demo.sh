#!/bin/bash -l 
#SBATCH --time=23:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=60GB 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p yaoyi 
#SBATCH --gres=gpu:a100:1 

module load python3 
module load gcc/13.1.0-mptekim 
source activate spot 

cd /home/yaoyi/lin00786/work/DeepLATTE/permafrost/spatial_prediction/MISO 

# python demo.py --grid_id AK050H48V07 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__c19__kfold0/ --interval 2 --output_tif AK050H48V07_pf1m_interval2.tif
python demo.py --grid_id AK050H48V07 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__c19__sfold0/ --interval 5 --output_tif AK050H48V07_pf1m_interval5.tif

# python demo.py --grid_id AK050H50V15 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_naive/sat_cov__c19__sfold0/ --interval 5 --output_tif AK050H50V15_pf1m_interval5.tif
# python demo.py --grid_id AK050H50V15 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_naive/sat_cov__c19__kfold0/ --interval 2 --output_tif AK050H50V15_pf1m_interval2.tif
# python demo.py --grid_id AK050H50V15 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_naive/sat_cov__c19__kfold0/ --interval 5 --output_tif AK050H50V15_pf1m_interval5.tif

# python test.py --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__uotr05__c19__tfold0
# python test.py --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__uotr05__c19__tfold1
# python test.py --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__uotr05__c19__tfold2
# python test.py --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__uotr05__c19__tfold3
# python test.py --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__uotr05__c19__tfold4

