# BLMFMADF

#The workflow of our proposed BLMFMADF model

![image](https://github.com/Mayingjun20179/BLMFMADF/blob/main/workflow.png)

#Environment Requirement

tensorly==0.8.1

torch==2.4.1+cu121

pandas==2.0.3

deepchem==2.8.0

rdkit==2022.9.4

networkx==2.8.8

torch-geometric==2.6.1

torch_scatter==2.1.2+pt24cu121

#Documentation

data/MDAD: Experimental data for baseline data MDAD

data/MASI: Experimental data for baseline data MASI

result_BLMFMADF_MDAD: After running the program, the location where the experimental result of the benchmark data MDAD is stored.

result_BLMFMADF_MASI: After running the program, the location where the experimental result of the benchmark data MASI is stored.

#Usage

First, install all the packages required by “requirements.txt”.

Second, run the program “Main_BLMFMADF.py” to get all the prediction results of BLMFMADF for the two benchmark datasets in the scenarios of ρ=1, ρ=0.1 (1/ρ=10) and ρ=0.02 (1/ρ=50).
