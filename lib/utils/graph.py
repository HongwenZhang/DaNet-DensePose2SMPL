import numpy as np
import torch

class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1,
                 norm_type='digraph'):
        self.max_hop = max_hop
        self.dilation = dilation
        self.norm_type = norm_type

        assert self.norm_type in ['digraph', 'undigraph', 'none']

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'smpl':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            # from modeling.smpl_heads import smpl_knowledge
            # neighbor_link = smpl_knowledge('limb_pairs')
            neighbor_link = [(0, 1), (1, 4), (4, 7), (7, 10),
                             (0, 2), (2, 5), (5, 8), (8, 11),
                             (0, 3), (3, 6), (6, 9),
                             (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
                             (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
                             (9, 12), (12, 15)]
                             # (12, 17), (12, 16)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'smpl_2neigh':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            # from modeling.smpl_heads import smpl_knowledge
            # neighbor_link = smpl_knowledge('limb_pairs')
            neighbor_link = [(0, 1), (1, 4), (4, 7), (7, 10),
                             (0, 2), (2, 5), (5, 8), (8, 11),
                             (0, 3), (3, 6), (6, 9),
                             (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
                             (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
                             (9, 12), (12, 15),
                             (12, 17), (12, 16)]
            neighbor2_link = [(0, 4), (0, 5), (0, 6),
                             (2, 8), (1, 7), (5, 11), (4, 10),
                             (3, 9), (6, 12), (9, 15),
                             (6, 13), (9, 16), (13, 18), (16, 20), (18, 22),
                             (6, 14), (9, 17), (14, 19), (17, 21), (19, 23)]
            self.edge = self_link + neighbor_link + neighbor2_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        if self.norm_type == 'digraph':
            normalize_adjacency = normalize_digraph(adjacency)
        elif self.norm_type == 'undigraph':
            normalize_adjacency = normalize_undigraph(adjacency)
        elif self.norm_type == 'none':
            normalize_adjacency = adjacency

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A, AD_mode=True):
    if type(A).__module__ == 'numpy':
        if AD_mode:
            Dl = np.sum(A, 0)
            num_node = A.shape[0]
            Dn = np.zeros((num_node, num_node))
            for i in range(num_node):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i]**(-1)
            AD = np.dot(A, Dn)
            return AD
        else:
            Dl = np.sum(A, 1)
            num_node = A.shape[0]
            Dn = np.zeros((num_node, num_node))
            for i in range(num_node):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i]**(-1)
            DA = np.dot(Dn, A)
            return DA
    elif torch.is_tensor(A):
        device = A.device
        if AD_mode:
            AD = []
            for bs in range(A.size(0)):
                A_bs = A[bs]
                Dl = torch.sum(A_bs, 0)
                num_node = A_bs.shape[0]
                Dn = torch.zeros((num_node, num_node)).to(device)
                for i in range(num_node):
                    if Dl[i] > 0:
                        Dn[i, i] = Dl[i] ** (-1)
                AD.append(torch.matmul(A, Dn))
            if len(AD) == 1:
                AD = AD[0].unsqueeze(0)
            else:
                AD = torch.stack(AD)
            return AD
        else:
            DA = []
            for bs in range(A.size(0)):
                A_bs = A[bs]
                Dl = torch.sum(A_bs, 1)
                num_node = A_bs.shape[0]
                Dn = torch.zeros((num_node, num_node)).to(device)
                for i in range(num_node):
                    if Dl[i] > 0:
                        Dn[i, i] = Dl[i] ** (-1)
                DA.append(torch.matmul(Dn, A))
            if len(DA) == 1:
                DA = DA[0].unsqueeze(0)
            else:
                DA = torch.stack(DA)
            return DA


def normalize_undigraph(A):

    if type(A).__module__ == 'numpy':
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
    elif torch.is_tensor(A):
        device = A.device
        DAD = []
        for bs in range(A.size(0)):
            A_bs = A[bs]
            Dl = torch.sum(A_bs, 0)
            num_node = A_bs.shape[0]
            Dn = torch.zeros((num_node, num_node)).to(device)
            for i in range(num_node):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i] ** (-0.5)
            DAD.append(torch.matmul(torch.matmul(Dn, A_bs), Dn))

        if len(DAD) == 1:
            DAD = DAD[0].unsqueeze(0)
        else:
            DAD = torch.stack(DAD)

        return DAD
