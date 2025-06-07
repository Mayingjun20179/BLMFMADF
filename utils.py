import torch
import numpy as np
import argparse
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import random
import os
# import matlab.engine
# eng = matlab.engine.start_matlab()
from ConstructHW import *
from sklearn.decomposition import PCA
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

def parse():
    p = argparse.ArgumentParser("VBMGDL: Variational Bayesian Inference with Hybrid Graph Deep Learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='DATA1_zhang', help='data name ')
    p.add_argument('--dataset', type=str, default='./DATA', help='dataset name')
    p.add_argument('--model-name', type=str, default='VBMGDL', help='VBMGDL')
    p.add_argument('--activation', type=str, default='tanh', help='activation layer between MGConvs')
    p.add_argument('--alpha_lambda', type=float, default=1, help='The alpha of lambda')
    p.add_argument('--beta_lambda', type=float, default=1, help='The beta of lambda')
    p.add_argument('--c', type=float, default=16, help='Importance level parameter')
    p.add_argument('--rank', type=int, default=50, help='rank')
    p.add_argument('--in_dim', type=int, default=50, help='Input the low-dimensional feature dimension')
    p.add_argument('--lr', type=float, default=0.001, help='learning rate')
    p.add_argument('--L2', type=float, default=1e-4, help='weight_decay')
    p.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    p.add_argument('--seed',type=float,default=1,help = 'The seed')
    p.add_argument('--nlayer',type=int,default=5,help = 'The number of layers of GCN and HGCN')
    p.add_argument('--K_neig', type=int, default=5, help='The number of neighbors in KNN')
    p.add_argument('--iteration', type=int, default=100, help='The number of iter to train')
    return p.parse_args()

def Const_hyper(args,sim_H,train_matrix):
    #
    concat_mic = torch.cat((train_matrix.T, sim_H), dim=1)
    H_mic_Kn,HGM = constructHW_knn(np.array(concat_mic), K_neigs=[args.K_neig], is_probH=False)
    args.HGM = HGM.to(args.device)
    #
    pca = PCA(n_components=args.in_dim)
    concat_mic = pca.fit_transform(np.array(concat_mic))
    concat_mic = torch.tensor(concat_mic,dtype=torch.float32)
    concat_mic = normalize_rows_zscore(concat_mic)
    args.H_feature = concat_mic.to(args.device)

    return args


def normalize_rows_zscore(x: torch.Tensor) -> torch.Tensor:

    #
    mean = torch.mean(x, dim=1, keepdim=True)
    std = torch.std(x, dim=1, keepdim=True)
    #
    std = torch.clamp(std, min=1e-12)
    #
    return (x - mean) / std

def top_sim(A, k=10):

    #
    result = torch.zeros_like(A)

    #
    values, indices = torch.topk(A, k, dim=1)

    #
    rows = torch.arange(A.size(0)).unsqueeze(1).expand(-1, k)

    #
    result[rows, indices] = values

    A_new = (result+result.T)/2

    return A_new






def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)





def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)  
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)  
    # the degree of the node
    DV = np.sum(H * W, axis=1)  
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  

    invDE = np.mat(np.diag(np.power(DE, -1)))  
    invDE[np.isinf(invDE)] = 0
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0
    W = np.mat(np.diag(W)) 
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr