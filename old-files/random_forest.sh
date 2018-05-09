#!/bin/bash
#SBATCH -p lrgmem 
#SBATCH -t 02:0:0

python random_forest.py

