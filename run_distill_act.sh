#!/bin/bash
#SBATCH -J distill_act
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/distill_act_%j.out
#SBATCH -e logs/distill_act_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tvisha_shah@brown.edu

module load python/3.12.4
module load cuda/12.2.0

cd ~/robo_self_improve
source venv/bin/activate

mkdir -p logs

python distill_act.py --n-demos 2000 --n-epochs 600
