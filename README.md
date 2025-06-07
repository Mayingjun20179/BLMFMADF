# BLMFMADF
Bayesian Logistic Matrix Factorization with Multi-Kernel Adaptive Deep Fusion for Microbe-Drug Association Prediction

Exploring potential microbe-drug associations (MDAs) not only facilitates drug discovery and clinical treatment but also contributes to a deeper understanding of microbial mechanisms. However, most MDA discoveries rely on biological experiments, which are time-consuming and costly. Therefore, developing an effective computational model to predict novel MDAs is of great importance. In this study, we propose a Bayesian Logistic Matrix Factorization model with Multi-kernel Adaptive Deep Fusion (BLMFMADF) for MDA prediction. We first integrate multi-omics data to construct drug molecular graphs and microbe hypergraph. Then, we employ Graph Convolutional Neural Network and Hypergraph Convolutional Neural Network to extract multi-level similarities of drugs and microbes, respectively. An attention mechanism is subsequently introduced to adaptively fuse these multi-level similarities, which are then incorporated into the Bayesian logistic matrix factorization framework to guide the generation of latent variable distributions. Additionally, we develop a variational Expectation-Maximization (EM) algorithm for adaptive inference of model hyperparameters and latent variables, which also guides the training of the deep learning model. Experimental results on two benchmark datasets across three scenarios show that, compared to other state-of-the-art methods, BLMFMADF achieves higher AUPR, AUC, and F1 scores in both balanced and highly imbalanced scenarios. Moreover, case studies further confirm that BLMFMADF can serve as an effective tool for MDA prediction.
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
