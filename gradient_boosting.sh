#!/bin/bash
#SBATCH -p lrgmem 
#SBATCH -t 02:0:0

python gradient_boosting.py

