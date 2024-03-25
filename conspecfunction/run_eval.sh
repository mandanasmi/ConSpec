#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=sbatch_out/conspec_key_to_door2_eval.%A.%a.out
#SBATCH --error=sbatch_err/conspec_key_to_door2_eval.%A.%a.err
#SBATCH --job-name=conspec_key_to_door2_eval

echo "Date:     $(date)"


module load anaconda/3
module load cuda/10.0/cudnn/7.6
conda activate py37tf15 

# Stage dataset into $SLURM_TMPDIR

python -u main_eval.py --pycolab_game key_to_door2 --num_episodes 5000 --seed 1 --num-processes 32 --checkpoint_interval 1 --start_checkpoint 0


