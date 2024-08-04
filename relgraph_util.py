from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import *
import numpy as np
import math
import igraph
from grapher import Grapher, IndexGraph
import argparse
import random
import igraph

def create_relation_graph(history_quads,num_ent,num_rel):
    ind_hs = [quads[:,:2] for quads in history_quads]
    ind_ts = [quads[:,1:3] for quads in history_quads]

    H_list = [csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel)) for ind_h in ind_hs]
    T_list = [csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel)) for ind_t in ind_ts]

    diag_vals_H = [H.sum(axis=1).A1 for H in H_list]
    for diag_vals_h in diag_vals_H:
        diag_vals_h[diag_vals_h!=0] = 1/diag_vals_h[diag_vals_h!=0]

    diag_vals_T = [T.sum(axis=1).A1 for T in T_list]
    for diag_vals_t in diag_vals_T:
        diag_vals_t[diag_vals_t!=0] = 1/diag_vals_t[diag_vals_t!=0]

    D_H_inv = [csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent)) for diag_vals_h in diag_vals_H]
    D_T_inv = [csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent)) for diag_vals_t in diag_vals_T]

    A_H = 0
    D_H_list = [D_H_inv[j] @ H_list[j] for j in range(len(H_list))]
    D_T_list = [D_T_inv[j] @ T_list[j] for j in range(len(T_list))]
    A_C = 0

    for i in range(len(H_list)):
        AT = [D_T_list[j].transpose() @ D_H_list[i] for j in range(i+1)]
        AT = np.sum(np.array(AT), axis=-1)
        A_C = A_C+AT
    #
    # for i in range(len(H_list)):
    #     H_H = D_H_list[i].transpose() @ D_H_list[i]
    #     H_H = H_H.todense()
    #     np.fill_diagonal(H_H,0)
    #     AD = [D_H_list[i].transpose() @ D_H_list[j] for j in range(i+1,len(H_list))]
    #     A_h = np.sum(np.array(AD),axis=-1)
    #     A_H = A_H+A_h +H_H
    #
    # A_H = np.concatenate((A_H[num_rel:,:],A_H[:num_rel,:]),axis=0)

    return A_C

def generate_relation_graph(history_quads,num_ent,num_rel):
    ind_hs = [quads[:,:2] for quads in history_quads]
    ind_ts = [quads[:, 1:3] for quads in history_quads]
    H_list = [csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel)) for ind_h in ind_hs]
    T_list = [csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel)) for ind_t in ind_ts]

    diag_vals_H = [H.sum(axis=1).A1 for H in H_list]
    for diag_vals_h in diag_vals_H:
        diag_vals_h[diag_vals_h!=0] = 1/diag_vals_h[diag_vals_h!=0]

    diag_vals_T = [T.sum(axis=1).A1 for T in T_list]
    for diag_vals_t in diag_vals_T:
        diag_vals_t[diag_vals_t!=0] = 1/diag_vals_t[diag_vals_t!=0]

    D_H_inv = [csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent)) for diag_vals_h in diag_vals_H]
    D_T_inv = [csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent)) for diag_vals_t in diag_vals_T]


    # diag_vals_H = [H.sum(axis=1).A1 for H in H_list]
    # for diag_vals_h in diag_vals_H:
    #     diag_vals_h[diag_vals_h!=0] = 1/diag_vals_h[diag_vals_h!=0]
    #
    # D_H_inv = [csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent)) for diag_vals_h in diag_vals_H]

    D_T_list = [D_T_inv[j] @ T_list[j] for j in range(len(T_list))]
    D_H = D_H_inv[-1] @ H_list[-1]
    # A_H = D_H_list[0].transpose() @ D_H_list[0]
    # A_H = A_H.todense()
    # np.fill_diagonal(A_H, 0)
    AD = [D_T_list[i].transpose() @ D_H for i in range(len(D_T_list))]
    A_h = np.sum(np.array(AD), axis=-1)
    # A_H = A_H + A_h
    #
    # A_H = np.concatenate((A_H[num_rel:,:],A_H[:num_rel,:]),axis=0)

    return A_h


def get_relation_triplets(G_rel):
    rel_triplets = []
    for tup in G_rel.get_edgelist():
        h, t = tup
        tupid = G_rel.get_eid(h, t)
        w = G_rel.es[tupid]["weight"]
        rel_triplets.append((int(h), int(t), float(w)))
    rel_triplets = np.array(rel_triplets)

    return rel_triplets