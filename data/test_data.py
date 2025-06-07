import pandas as pd
import numpy as np


drug_sim = pd.read_csv('MASI/drug_sim_MASI.csv',index_col=0)
mic_sim = pd.read_csv('MASI/micro_sim_MASI.csv',index_col=0)
mic_drug = pd.read_csv('MASI/micro_drug_MASI.csv')
mic_sim.index = mic_sim.columns
drug_sim.index = drug_sim.columns

#
drug_size = 100
random_indices = np.random.choice(drug_sim.shape[0], size=drug_size, replace=False)
drug_sim = drug_sim.iloc[random_indices, random_indices]# 根据随机索引采样行和列

mic_size = 50
random_indices = np.random.choice(mic_sim.shape[0], size=mic_size, replace=False)
mic_sim = mic_sim.iloc[random_indices, random_indices] # 根据随机索引采样行和列


mic_drug = mic_drug[mic_drug['pubchem_id'].astype(str).isin(drug_sim.columns)]
mic_drug = mic_drug[mic_drug['micro_tid'].astype(str).isin(mic_sim.columns)]

#
matching_indices_drug = drug_sim.columns.intersection(mic_drug['pubchem_id'].astype(str).unique())
matching_indices_mic = mic_sim.columns.intersection(mic_drug['micro_tid'].astype(str).unique())

drug_sim = drug_sim.loc[matching_indices_drug,matching_indices_drug]
mic_sim = mic_sim.loc[matching_indices_mic,matching_indices_mic]

drug_sim.to_csv('MASI1/drug_sim_MASI.csv',index=True)
mic_sim.to_csv('MASI1/micro_sim_MASI.csv',index=True)
mic_drug.to_csv('MASI1/micro_drug_MASI.csv',index=False)

drug_inf = pd.read_csv('MASI/drug_inf_MASI.csv')
drug_inf['pubchem_id'] = drug_inf['pubchem_id'].astype(str)
drug_inf = drug_inf.set_index('pubchem_id').loc[matching_indices_drug].reset_index()
drug_inf = drug_inf.rename(columns={'index': 'pubchem_id'})
drug_inf.to_csv('MASI1/drug_inf_MASI.csv',index=False)