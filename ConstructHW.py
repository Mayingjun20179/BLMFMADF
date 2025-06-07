import numpy as np
import torch
from sklearn.cluster import KMeans


def constructHW_knn(X,K_neigs,is_probH):
    # hc_KNN = hypergraph_construct_KNN1()
    # """incidence matrix"""
    # H,W = hc_KNN.construct_H_with_KNN(X, K_neigs, is_probH=is_probH)
    # G = hc_KNN._generate_G_from_H(H,W)

    hc_KNN = hypergraph_construct_KNN()
    """incidence matrix"""
    H = hc_KNN.construct_H_with_KNN(X, K_neigs, is_probH=is_probH)
    G = hc_KNN._generate_G_from_H(H)

    return H,G

def constructHW_kmean(X,clusters):

    """incidence matrix"""
    hc_kmeans = hypergraph_construct_kmeans()
    H = hc_kmeans.construct_H_with_Kmeans(X, clusters)

    G = hc_kmeans._generate_G_from_H(H)

    return H,G


class hypergraph_construct_KNN:
    def __init__(self):
        print("hypergraph construct KNN")

    def Eu_dis(self,x):

        x = np.mat(x)

        aa = np.sum(np.multiply(x, x), 1)
        ab = x * x.T
        dist_mat = aa + aa.T - 2 * ab
        dist_mat[dist_mat < 0] = 0

        dist_mat = np.sqrt(dist_mat)
        dist_mat = np.maximum(dist_mat, dist_mat.T)
        return dist_mat

    def feature_concat(self,*F_list, normal_col=False):

        features = None
        for f in F_list:
            if f is not None and f != []:

                if len(f.shape) > 2:
                    f = f.reshape(-1, f.shape[-1])

                if normal_col:
                    f_max = np.max(np.abs(f), axis=0)
                    f = f / f_max

                if features is None:
                    features = f
                else:
                    features = np.hstack((features, f))
        if normal_col:
            features_max = np.max(np.abs(features), axis=0)
            features = features / features_max
        return features

    def hyperedge_concat(self,*H_list):

        H = None
        for h in H_list:
            # if h is not None and h != []:
            if h is not None and len(h) != 0:
                # for the first H appended to fused hypergraph incidence matrix
                if H is None:
                    H = h
                else:
                    if type(h) != list:
                        H = np.hstack((H, h))
                    else:
                        tmp = []
                        for a, b in zip(H, h):
                            tmp.append(np.hstack((a, b)))
                        H = tmp
        return H

    def generate_G_from_H(self,H, variable_weight=False):

        if type(H) != list:
            return self._generate_G_from_H(H, variable_weight)
        else:
            G = []
            for sub_H in H:
                G.append(self.generate_G_from_H(sub_H, variable_weight))
            return G

    def _generate_G_from_H(self,H, variable_weight=False):

        H = np.array(H)

        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)

        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            G = torch.Tensor(G)
            return G

    def construct_H_with_KNN_from_distance(self,dis_mat, k_neig, is_probH=False, m_prob=1):

        n_obj = dis_mat.shape[0]
        n_edge = n_obj
        H = np.zeros((n_obj, n_edge))
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(
                        -dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  ## affinity matrix的计算公式
                else:
                    H[node_idx, center_idx] = 1.0
        return H

    def construct_H_with_KNN(self,X, K_neigs, split_diff_scale=False, is_probH=False, m_prob=1):

        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[-1])

        if type(K_neigs) == int:
            K_neigs = [K_neigs]

        dis_mat = self.Eu_dis(X)
        H = []
        for k_neig in K_neigs:
            H_tmp = self.construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)

            if not split_diff_scale:
                H = self.hyperedge_concat(H, H_tmp)
            else:
                H.append(H_tmp)

        return H



class hypergraph_construct_KNN1:
    def __init__(self):
        print("hypergraph construct KNN")

    def Eu_dis(self,x):

        x = np.mat(x)

        aa = np.sum(np.multiply(x, x), 1)
        ab = x * x.T
        dist_mat = aa + aa.T - 2 * ab
        dist_mat[dist_mat < 0] = 0

        dist_mat = np.sqrt(dist_mat)
        dist_mat = np.maximum(dist_mat, dist_mat.T)
        return dist_mat

    def feature_concat(self,*F_list, normal_col=False):

        features = None
        for f in F_list:
            if f is not None and f != []:

                if len(f.shape) > 2:
                    f = f.reshape(-1, f.shape[-1])

                if normal_col:
                    f_max = np.max(np.abs(f), axis=0)
                    f = f / f_max

                if features is None:
                    features = f
                else:
                    features = np.hstack((features, f))
        if normal_col:
            features_max = np.max(np.abs(features), axis=0)
            features = features / features_max
        return features

    def hyperedge_concat(self,*H_list):

        H = None
        for h in H_list:
            # if h is not None and h != []:
            if h is not None and len(h) != 0:
                # for the first H appended to fused hypergraph incidence matrix
                if H is None:
                    H = h
                else:
                    if type(h) != list:
                        H = np.hstack((H, h))
                    else:
                        tmp = []
                        for a, b in zip(H, h):
                            tmp.append(np.hstack((a, b)))
                        H = tmp
        return H

    def generate_G_from_H(self,H, variable_weight=False):

        if type(H) != list:
            return self._generate_G_from_H(H, variable_weight)
        else:
            G = []
            for sub_H in H:
                G.append(self.generate_G_from_H(sub_H, variable_weight))
            return G

    def _generate_G_from_H(self,H,W, variable_weight=False):

        H = np.array(H)

        n_edge = H.shape[1]
        # the weight of the hyperedge

        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            G = torch.Tensor(G)
            return G

    def construct_H_with_KNN_from_distance(self,dis_mat, k_neig, is_probH=False, m_prob=1):

        n_obj = dis_mat.shape[0]
        n_edge = n_obj
        H = np.zeros((n_obj, n_edge))
        W = np.zeros(n_edge)
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(
                        -dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  ## affinity matrix的计算公式
                else:
                    H[node_idx, center_idx] = 1.0
            W[center_idx] = np.average(dis_vec[0, nearest_idx])
        return H,W

    def construct_H_with_KNN(self,X, K_neigs,split_diff_scale=False, is_probH=False, m_prob=1):

        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[-1])

        if type(K_neigs) == int:
            K_neigs = [K_neigs]

        dis_mat = self.Eu_dis(X)
        H = []  #这个是把多个超图拼起来
        for k_neig in K_neigs:
            H_tmp,W = self.construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)

            if not split_diff_scale:
                H = self.hyperedge_concat(H, H_tmp)
            else:
                H.append(H_tmp)

        return H,W


class hypergraph_construct_kmeans:
    def __init__(self):
        print("hypergraph construct kmeans")

    def hyperedge_concat(self,*H_list):

        H = None
        for h in H_list:
            # if h is not None and h != []:
            if h is not None and len(h) != 0:
                # for the first H appended to fused hypergraph incidence matrix
                if H is None:
                    H = h
                else:
                    if type(h) != list:
                        H = np.hstack((H, h))
                    else:
                        tmp = []
                        for a, b in zip(H, h):
                            tmp.append(np.hstack((a, b)))
                        H = tmp
        return H

    def _construct_edge_list_from_cluster(self,X, clusters):
        """
        construct edge list (numpy array) from cluster for single modality
        :param X: feature
        :param clusters: number of clusters for k-means
        :param adjacent_clusters: a node's adjacent clusters
        :param k_neighbors: number of a node's neighbors
        :return:
        """
        N = X.shape[0]
        kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=0).fit(X)

        assignment = kmeans.labels_

        H = np.zeros([N, clusters])

        for i in range(N):
            H[i, assignment[i]] = 1

        return H

    def construct_H_with_Kmeans(self,X, clusters, split_diff_scale=False):

        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[-1])

        if type(clusters) == int:
            clusters = [clusters]

        H = []
        for clusters in clusters:
            H_tmp = self._construct_edge_list_from_cluster(X, clusters)

            if not split_diff_scale:
                H = self.hyperedge_concat(H, H_tmp)
            else:
                H.append(H_tmp)
        return H

    def _generate_G_from_H(self,H, variable_weight=False):

        H = np.array(H)

        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)

        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            G = torch.Tensor(G)
            return G

    def generate_G_from_H(self,H, variable_weight=False):

        if type(H) != list:
            return self._generate_G_from_H(H, variable_weight)
        else:
            G = []
            for sub_H in H:
                G.append(self.generate_G_from_H(sub_H, variable_weight))
            return G