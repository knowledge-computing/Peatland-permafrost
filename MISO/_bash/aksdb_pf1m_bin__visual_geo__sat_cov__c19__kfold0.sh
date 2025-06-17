#!/bin/bash -l 
#SBATCH --time=23:00:00 
#SBATCH --ntasks=8 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p msigpu 
#SBATCH --gres=gpu:1 

module load python3 
module load gcc/13.1.0-mptekim 
source activate spot 

cd /home/yaoyi/lin00786/work/DeepLATTE/permafrost/spatial_prediction/MISO 

CUDA_VISIBLE_DEVICES="0" python train.py --config configs/aksdb_pf1m_bin/visual_geo__sat_cov__c19.yaml --local_rank 0 --master_port 18843 --fold_id 0 --split_mode kfold --split_file data/kfold5_split_train_test_indices.json