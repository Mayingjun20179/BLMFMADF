###默认两层
import numpy as np
import tensorly as tl
# tl.set_backend('numpy')
import torch
from BLMFMADF_GCN import Model
from utils import *
from get_data import GetData
from Evaluate import cv_tensor_model_evaluate
import pandas as pd


class Experiments(object):
    def __init__(self, GH_data, model_name='BLMFMADF', **kwargs):
        super().__init__()
        self.GH_data = GH_data
        self.model_name = model_name
        self.parameters = kwargs

    def CV_asso(self, args, prop):
        k_folds = 5
        index_matrix = np.array(np.where(self.GH_data.adj_matrix == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)
        np.random.seed(args.seed)
        np.random.shuffle(index_matrix.T)

        metrics_matrix = np.zeros((1, 7))
        result_100 = []
        for k in range(k_folds):

            train_matrix = np.array(self.GH_data.adj_matrix, copy=True)
            if k != k_folds - 1:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_matrix[test_index] = 0
            train_matrix = torch.tensor(train_matrix, dtype=torch.float32)

            ###Constructing Hybrid graphs
            args = Const_hyper(args, self.GH_data.mic_sim,train_matrix)
            # Predictions of the BLMFMADF model
            self.model = Model(args, self.model_name)
            pre_score, _ = self.model.BLMFMADF_opt(train_matrix)
            predict_matrix = np.array(pre_score)
            # Randomized Generation of Test Negatives and Model Evaluation
            for i in range(1,20000,1000):
                jieguo = cv_tensor_model_evaluate(self.GH_data, predict_matrix, test_index, i, prop)
                metrics_matrix = metrics_matrix + jieguo
                result_100.append(jieguo)
            print(np.array(result_100).mean(axis=0))

        # Get result
        result = pd.DataFrame(np.around(metrics_matrix / 100, decimals=4)[:, 0:3], columns=['AUPR', 'AUC', 'F1'])
        result_100 = pd.DataFrame(np.array(result_100)[:, 0:3], columns=['AUPR', 'AUC', 'F1'])
        print(result)
        return result, result_100


if __name__ == '__main__':
    #
    args = parse()
    args.device = torch.device('cuda:0')

    #####
    seed = 1
    set_seed(seed)
    root = './data/MASI'
    GH_data = GetData(root)
    args.durg_inf = GH_data.batch_drug.to(args.device)
    args.use_GMP = True
    args.G_num, args.H_num = GH_data.N_drug, GH_data.N_mic
    experiment = Experiments(GH_data, model_name='BLMFMADF')

    # CV_asso
    prop = [1, 10, 50]
    args.asso = False
    for kk in prop:
        result_CV_asso, result_CV_asso100 = experiment.CV_asso(args, kk)
        file_path = './result_BLMFMADF_MASI/BLMFMADF_asso' + '_prop_' + str(kk)  + '.txt'
        result_CV_asso.to_csv(file_path, index=False, sep='\t')
        print(result_CV_asso)


    #####
    seed = 1
    set_seed(seed)
    root = './data/MDAD'
    GH_data = GetData(root)
    args.durg_inf = GH_data.batch_drug.to(args.device)
    args.use_GMP = True
    args.G_num, args.H_num = GH_data.N_drug, GH_data.N_mic
    experiment = Experiments(GH_data, model_name='BLMFMADF')

    # # CV_asso
    prop = [1, 10,50]

    for kk in prop:
        result_CV_asso, result_CV_asso100 = experiment.CV_asso(args, kk)
        print(result_CV_asso)
        file_path = './result_BLMFMADF_MDAD/BLMFMADF_asso' + '_prop_' + str(kk) + '_nlayer_5'+ '.txt'
        result_CV_asso.to_csv(file_path, index=False, sep='\t')


