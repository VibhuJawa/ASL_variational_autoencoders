#!/bin/bash
#SBATCH -n 4 
#SBATCH -p lrgmem 
#SBATCH -t 08:0:0

python main.py

