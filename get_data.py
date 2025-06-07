import os.path as osp
import pandas as pd
from process_smiles import *


class GetData(object):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.batch_drug, self.mic_sim,self.adj_matrix,self.index_0,self.N_0 = self.__get_data__()
        self.N_drug, self.N_mic = self.adj_matrix.shape


    def __get_data__(self):
        if 'MASI' in self.root:
            smiles_file = osp.join(self.root,'drug_inf_MASI.csv')
            drug_inf,batch_drug = drug_fea_process(smiles_file)
            mic_file = osp.join(self.root,'micro_sim_MASI.csv')
            mic_sim = pd.read_csv(mic_file,index_col=0)
            adj_file = osp.join(self.root, 'micro_drug_MASI.csv')
            adj_inf = pd.read_csv(adj_file)
        else:
            smiles_file = osp.join(self.root,'drug_inf_MDAD.csv')
            drug_inf,batch_drug = drug_fea_process(smiles_file)
            mic_file = osp.join(self.root,'micro_sim_MDAD.csv')
            mic_sim = pd.read_csv(mic_file,index_col=0)
            adj_file = osp.join(self.root, 'micro_drug_MDAD.csv')
            adj_inf = pd.read_csv(adj_file)

        #构建关联矩阵
        drug_num = drug_inf.shape[0]
        d_map = dict(zip(drug_inf['drug_name'], range(drug_num)))

        mic_num = mic_sim.shape[0]
        m_map = dict(zip(mic_sim.columns, range(mic_num)))
        adj_list = [[d_map[str(adj_inf.iloc[i,2])],m_map[str(adj_inf.iloc[i,1])]] for i in range(adj_inf.shape[0])]
        adj_ind = np.array(adj_list,dtype=np.int64)
        adj_ind = torch.from_numpy(adj_ind)
        adj_matrix = torch.zeros(drug_num,mic_num)
        adj_matrix[adj_ind[:,0],adj_ind[:,1]]=1
        #
        mic_sim = torch.from_numpy(np.array(mic_sim)).type(torch.float32)

        index_0 = np.array(np.where(adj_matrix.numpy() == 0)).T
        N_0 = index_0.shape[0]

        return batch_drug,mic_sim,adj_matrix,index_0,N_0

def drug_fea_process(smiles_file):
    drug_inf = pd.read_csv(smiles_file)
    smile_graph = []
    for i in range(drug_inf.shape[0]):
        smile = drug_inf.iloc[i,2]
        g = smile_to_graph(smile) #
        smile_graph.append(g)
    drug_num = len(smile_graph)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False,
                                           collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_inf,drug_data