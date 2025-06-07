import torch
import numpy as np
from torch import nn
from torch.linalg import inv
import tensorly as tl
from tensorly.tenalg import khatri_rao
tl.set_backend('pytorch')
from F_mlayer_model import *
from utils import top_sim

class Model(object):
    def __init__(self,args,name='BLMFMADF',**kwargs):
        super().__init__()
        self.name = name
        self.args = args
        self.paramater = kwargs
        self.model = Hybridgraphattention(HgnnEncoder(args = args),
            GcnEncoder(args = args),args).to(args.device)

    def jisuan_lamb(self,kx):
        sig_kx = 1. / (1 + torch.exp(-kx))
        lam_kx = 1. / (2 * kx) * (sig_kx - 1 / 2)
        return lam_kx, sig_kx

    def BLMFMADF_opt(self, Y):

        Y = torch.tensor(Y,dtype=torch.float32)

        args = self.args

        I, J = Y.shape
        R = args.rank

        # Set random seed
        torch.manual_seed(args.seed)

        # Initialize G，H，U，V
        G_mu = torch.randn(I, R,dtype=torch.float32)
        G_sigma = torch.tile(torch.eye(R,dtype=torch.float32).unsqueeze(0), (I, 1, 1))

        H_mu = torch.randn(J, R,dtype=torch.float32)
        H_sigma = torch.tile(torch.eye(R,dtype=torch.float32).unsqueeze(0), (J, 1, 1))

        U_mu = torch.randn(I, R,dtype=torch.float32)
        U_sigma = torch.tile(torch.eye(I,dtype=torch.float32).unsqueeze(0), (R, 1, 1))

        V_mu = torch.randn(J, R,dtype=torch.float32)
        V_sigma = torch.tile(torch.eye(J,dtype=torch.float32).unsqueeze(0), (R, 1, 1))

        #
        Bg = G_sigma.reshape(I, R * R)
        Bh = H_sigma.reshape(J, R * R)

        # Initialize Lambda
        Lambda_alpha = (args.alpha_lambda + (I + J) / 2) * torch.ones(R,dtype=torch.float32)
        Lambda_beta = args.beta_lambda * torch.ones(R,dtype=torch.float32)

        # Precompute some matrices
        kx = torch.ones(I, J)
        lambdag_indices = torch.eye(R).repeat(I, 1, 1).bool()  # M x R x R
        lambdah_indices = torch.eye(R).repeat(J, 1, 1).bool()  # N x R x R

        c = args.c
        Aij = (c * Y - 1 + Y) / 2
        lam_kx, _ = self.jisuan_lamb(kx)
        Bij = 2 * (c * Y + 1 - Y) * lam_kx

        tao_g, tao_h = 1.0, 1.0
        #深度学习来获取本质相似性矩阵
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.L2)
        GH = torch.cat((G_mu, H_mu), dim=0).to(args.device)
        UV = torch.cat((U_mu, V_mu), dim=0).to(args.device)
        SU_sigma = U_sigma.sum(dim=0).to(args.device)
        SV_sigma = V_sigma.sum(dim=0).to(args.device)
        for epoch in range(args.epochs):
            self.model.train()
            optimizer.zero_grad()
            G_sim, H_sim = self.model(args)
            loss = Loss_fun_opt(GH, UV, G_sim,H_sim,tao_g,tao_h,SU_sigma,SV_sigma,args)
            # if epoch % 10 == 0:
            #     print(f'当前epoch为{epoch}，loss为{loss.item()}')
            loss.backward()
            optimizer.step()

        # Ku = torch.tensor(top_sim(G_sim.clone().detach().to('cpu')),dtype=torch.float32)
        # Kv = torch.tensor(top_sim(H_sim.clone().detach().to('cpu')),dtype=torch.float32)

        Ku = torch.tensor(G_sim.clone().detach().to('cpu'),dtype=torch.float32)
        Kv = torch.tensor(H_sim.clone().detach().to('cpu'),dtype=torch.float32)

        KuKu = Ku.T @ Ku
        KvKv = Kv.T @ Kv

        for iter in range(args.iteration):
            if iter % 10 == 0:
                print('.', end='')
            if iter % 50 == 0:
                print(f' {iter:5d}')

            # Update tao_g
            tao_g_a = (I * R) / 2
            tao_g_b = torch.trace((G_mu - Ku @ U_mu) @ (G_mu - Ku @ U_mu).T)+\
                        G_sigma[lambdag_indices].sum()
            for r in range(R):
                tao_g_b += torch.trace(Ku @ U_sigma[r,:, :] @ Ku.T)
            tao_g_b /= 2
            tao_g = tao_g_a / tao_g_b

            # Update tao_h
            tao_h_a = (J * R) / 2
            tao_h_b = torch.trace((H_mu - Kv @ V_mu) @ (H_mu - Kv @ V_mu).T)+\
                        H_sigma[lambdah_indices].sum()
            for r in range(R):
                tao_h_b += torch.trace(Kv @ V_sigma[r,:, :] @ Kv.T)
            tao_h_b /= 2
            tao_h = tao_h_a / tao_h_b

            # Update Lambda
            for r in range(R):
                Lambda_beta[r] = args.beta_lambda + 0.5 * (
                    U_mu[:, r] @ U_mu[:, r] + torch.trace(U_sigma[r,:, :]) +
                    V_mu[:, r] @ V_mu[:, r] + torch.trace(V_sigma[r,:, :]))
            Lambdas = Lambda_alpha/Lambda_beta


            ##### Update U,V
            for r in range(R):
                # U_sigma[r,:, :] = torch.pinverse(KuKu * tao_g + Lambdas[r] * torch.eye(I))
                U_sigma[r, :, :] = torch.linalg.inv(KuKu * tao_g + Lambdas[r] * torch.eye(I,dtype=torch.float32))
                U_mu[:, r] = U_sigma[r, :, :] @ Ku.T @ G_mu[:, r] * tao_g

            # Update V
            for r in range(R):
                V_sigma[r,:, :] = torch.linalg.inv(KvKv * tao_h + Lambdas[r] * torch.eye(J,dtype=torch.float32))
                V_mu[:, r] = V_sigma[r,:, :] @ Kv.T @ H_mu[:, r] * tao_h

            # Update G
            ENZZT = (Bij @ Bh).reshape(I, R, R)
            FslashY = H_mu.T @ Aij.T
            for i in range(I):
                G_sigma[i, :, :] = torch.linalg.inv(2 * ENZZT[i, :, :] + tao_g * torch.eye(R,dtype=torch.float32))
                G_mu[i, :] = G_sigma[i, :, :] @ (FslashY[:, i] + tao_g * U_mu.T @ Ku[i])
            Bg = G_sigma.reshape(I, R * R) + khatri_rao([G_mu.T, G_mu.T]).T


            # Update H
            ENZZT = (Bij.T @ Bg).reshape(J, R, R)
            FslashY = G_mu.T @ Aij
            for j in range(J):
                H_sigma[j, :, :] = torch.linalg.inv(2 * ENZZT[j, :, :] + tao_h * torch.eye(R,dtype=torch.float32))  # Posterior covariance matrix
                H_mu[j, :] = H_sigma[j, :, :] @ (FslashY[:, j] + tao_h * V_mu.T @ Kv[j])  # Posterior expectation
            Bh = H_sigma.reshape(J, R * R) + khatri_rao([H_mu.T, H_mu.T]).T



            # Update kx
            kx2 = Bg @ Bh.T
            if (kx2 < 0).any():
                print(iter)
                raise ValueError('error')

            kx = torch.sqrt(kx2)
            lam_kx,_ = self.jisuan_lamb(kx)
            Bij = 2 * (c * Y + 1 - Y) * lam_kx

            if iter % 20 ==0:

                # update G_Z and H_Z
                GH = torch.cat((G_mu, H_mu), dim=0).to(args.device)
                UV = torch.cat((U_mu, V_mu), dim=0).to(args.device)
                SU_sigma = U_sigma.sum(dim=0).to(args.device)
                SV_sigma = V_sigma.sum(dim=0).to(args.device)
                for epoch in range(args.epochs):
                    self.model.train()
                    optimizer.zero_grad()
                    G_sim, H_sim = self.model(args)
                    loss = Loss_fun_opt(GH, UV, G_sim, H_sim, tao_g,tao_h,SU_sigma,SV_sigma,args)
                    # if epoch % 10 ==0:
                    #     print(f'当前epoch为{epoch}，loss为{loss.item()}')
                    loss.backward()
                    optimizer.step()

                # Ku = torch.tensor(top_sim(G_sim.clone().detach().to('cpu')),dtype=torch.float32)
                # Kv = torch.tensor(top_sim(H_sim.clone().detach().to('cpu')),dtype=torch.float32)

                Ku = torch.tensor(G_sim.clone().detach().to('cpu'), dtype=torch.float32)
                Kv = torch.tensor(H_sim.clone().detach().to('cpu'), dtype=torch.float32)

                KuKu = Ku.T @ Ku
                KvKv = Kv.T @ Kv

        # Prepare final score
        score = 1 / (1 + torch.exp(-G_mu @ H_mu.T))

        state = {
            'Lambda': {'alpha': Lambda_alpha, 'beta': Lambda_beta},
            'G': {'mu': G_mu, 'sigma': G_sigma},
            'H': {'mu': H_mu, 'sigma': H_sigma},
            'U': {'mu': U_mu, 'sigma': U_sigma},
            'V': {'mu': V_mu, 'sigma': V_sigma},
        }

        return score, state