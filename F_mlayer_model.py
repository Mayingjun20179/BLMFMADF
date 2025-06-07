import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,HypergraphConv, global_max_pool, global_mean_pool
from utils import reset
import tensorly as tl
from similarity import get_Gauss_Similarity_torch
tl.set_backend('pytorch')
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def Loss_fun_opt(GH, UV, G_sim,H_sim,tao_g,tao_h,SU_sigma,SV_sigma,args):
    Ng, Nh = args.G_num, args.H_num
    FG_emb, FH_emb = G_sim @ UV[:Ng], H_sim @ UV[Ng:]
    G_emb, H_emb = GH[:Ng], GH[Ng:]
    loss_g = torch.trace((G_emb - FG_emb).t() @ (G_emb-FG_emb))
    loss_g = loss_g + torch.trace(G_sim @ SU_sigma @ G_sim.T)
    loss_h = torch.trace((H_emb - FH_emb).t() @ (H_emb - FH_emb))
    loss_h = loss_h + torch.trace(H_sim @ SV_sigma @ H_sim.T)
    return tao_g*loss_g+tao_h*loss_h

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout

        self.hgnn1 = HGNN_conv(in_dim, hidden_list)

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        # x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed



class HgnnEncoder(torch.nn.Module):
    def __init__(self,args):
        super(HgnnEncoder, self).__init__()

        # # mic_knn
        in_dim = args.in_dim
        out_dim = args.rank
        self.conv_H_mic_knn = nn.ModuleList([HGCN(in_dim, out_dim)])
        self.batch_H_mic_knn = nn.ModuleList([nn.BatchNorm1d(out_dim)])
        for i in range(args.nlayer - 1):
            in_dim = out_dim
            out_dim = args.rank
            self.conv_H_mic_knn.append(HGCN(in_dim, out_dim))
            self.batch_H_mic_knn.append(nn.BatchNorm1d(out_dim))

        self.act = nn.Tanh()

    def forward(self,H_fea,args):

        #mic_knn
        x_H_mic_knn = []
        x = H_fea
        for i in range(args.nlayer):
            xi = self.batch_H_mic_knn[i](self.act(self.conv_H_mic_knn[i](x, args.HGM)))
            x = xi
            x_H_mic_knn.append(xi)
        return x_H_mic_knn

class GcnEncoder(nn.Module):
    def __init__(self,args):
        super(GcnEncoder, self).__init__()


        #-------G_layer  （drug）
        self.drug_inf = args.durg_inf
        dim_G = self.drug_inf.x.shape[1]
        self.use_GMP = args.use_GMP
        self.conv_g = nn.ModuleList([GCNConv(dim_G, args.rank)])
        self.batch_g = nn.ModuleList([nn.BatchNorm1d(args.rank)])
        for i in range(args.nlayer-1):
            self.conv_g.append(GCNConv(args.rank, args.rank))
            self.batch_g.append(nn.BatchNorm1d(args.rank))


        self.act = nn.Tanh()

    def forward(self, args):

        G_feature, edge_G, batch_G = self.drug_inf.x, self.drug_inf.edge_index, self.drug_inf.batch
        # -----G_train   (drug)
        x_G = []
        for i in range(args.nlayer):
            x_Gi = self.batch_g[i](self.act(self.conv_g[i](G_feature, edge_G)))
            G_feature = x_Gi
            if self.use_GMP:
                x_Gii = global_max_pool(x_Gi, batch_G)
                x_G.append(x_Gii)
            else:
                x_Gii = global_mean_pool(x_Gi, batch_G)
                x_G.append(x_Gii)

        return x_G

class Hybridgraphattention(torch.nn.Module):
    def __init__(self, hgcn_encoder,gcn_encoder,args):
        super(Hybridgraphattention, self).__init__()
        self.hgcn_encoder = hgcn_encoder
        self.gcn_encoder = gcn_encoder

        N_fea = args.nlayer
        self.G_weight = nn.Parameter(torch.ones(N_fea))
        self.H_weight = nn.Parameter(torch.ones(N_fea))

        self.reset_parameters()

        self.act = nn.ReLU()
    def reset_parameters(self):
        reset(self.gcn_encoder)
        reset(self.hgcn_encoder)

    def forward(self,args):
        # Ng,Nh = args.G_num,args.H_num
        # R = args.rank

        # torch.manual_seed(1)
        # H_fea = torch.randn(Nh, R).to(args.device)
        H_fea = args.H_feature
        G_emb,H_emb = [],[]

        #True:GNN
        # #True:GNN
        x_G = self.gcn_encoder(args)
        for i in range(args.nlayer):
            G_emb.append(x_G[i])

        x_Hh = self.hgcn_encoder(H_fea,args)
        for i in range(args.nlayer):
            H_emb.append(x_Hh[i])

        #step3:计算所有特征的高斯相似性
        S_G,S_H = [],[]
        for i in range(args.nlayer):
            S_G.append(get_Gauss_Similarity_torch(G_emb[i],'row'))
        for i in range(args.nlayer):
            S_H.append(get_Gauss_Similarity_torch(H_emb[i],'row'))


        G_weight = self.act(self.G_weight)/(self.act(self.G_weight).sum())


        H_weight = self.act(self.H_weight)/(self.act(self.H_weight).sum())

        G_sim_all = torch.stack([S_G[i] * G_weight[i] for i in range(len(S_G))]).sum(dim=0)
        H_sim_all = torch.stack([S_H[i] * H_weight[i] for i in range(len(S_H))]).sum(dim=0)

        # G_sim_all,H_sim_all = top_sim(G_sim_all),top_sim(H_sim_all)

        return G_sim_all,H_sim_all