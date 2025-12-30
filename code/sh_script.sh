#!/bin/bash
#SBATCH --job-name=trojanflow_implementation
#SBATCH --nodes=1
#SBATCH --partition=a10g-8-gm192-c192-m768
#SBATCH --output=log/%x-%j_embeddings.out
#SBATCH --error=log/%x-%j_embeddings.err
#SBATCH --gpus=1
#SBATCH --time=100:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --mail-user=lovelyyeswanth2002@gmail.com

python trojanflow_implementation.py