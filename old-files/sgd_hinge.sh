#!/bin/bash
#SBATCH -p lrgmem 
#SBATCH -t 02:0:0

python sgd_hinge.py

