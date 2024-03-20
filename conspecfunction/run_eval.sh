#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8GB
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_out/Conspec_k3_eval.%A.%a.out
#SBATCH --error=sbatch_err/Conspec_k3_eval.%A.%a.err
#SBATCH --job-name=Conspec_k3_eval

module load anaconda/3
module load cuda/10.0/cudnn/7.6
conda activate py37tf15 

python -u main_eval.py --pycolab_game key_to_door3 --num_episodes 2500 --seed 1 --num-processes 32 --checkpoint_interval 1 --start_checkpoint 0


