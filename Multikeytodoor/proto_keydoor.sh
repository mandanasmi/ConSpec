#!/usr/bin/env bash
#SBATCH --array=0-8%12
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:rtx8000:1
#####SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/proto_v1_test.%A.%a.out
#SBATCH --error=sbatch_err/proto_v1_test.%A.%a.err
#SBATCH --job-name=proto_v1_test


module load anaconda/3
module load cuda/10.0/cudnn/7.6
conda activate py37tf15 
python -u main.py --algo ppoConSpec --factorC 1. --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed 80001 --expansion 5000 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 0.2
