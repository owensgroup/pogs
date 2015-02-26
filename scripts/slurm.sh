#!/bin/bash
#
#SBATCH --job-name=pogs
#SBATCH --output=pogs.txt
#
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:g
