# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle



def make_adjacency(num_nodes):
    adj = np.zeros((num_nodes, num_nodes))

    s, clique_numbers = 5, list()

    for i in range(num_nodes):
        clique_numbers.append(s)
        if i % 2 == 1: s = s + 10
        else: s = s + 20
        if s > num_nodes: break

    # clique_numbers = [5, 25, 35, 55, 65, 85, 95, 115, 125, 145, 155, 175, 185]
    # print(clique_numbers)

    for i in range(adj.shape[0]):
        # Create connections till 4-hop on every 10th node (forward)
        if (i % 10 == 0) and (i+4 < adj.shape[1]): 
            adj[i][i+2], adj[i][i+3], adj[i][i+4] = 1, 1, 1
            adj[i+2][i], adj[i+3][i], adj[i+4][i] = 1, 1, 1

        # Create connections till 4-hop on every 10th node (backward)
        if (i % 10 == 0) and (i-4 > 0): 
            adj[i][i-2], adj[i][i-3], adj[i][i-4] = 1, 1, 1
            adj[i-2][i], adj[i-3][i], adj[i-4][i] = 1, 1, 1

        # Create 4-cliques in 15th, 45th, 75th node (forward)
        if (i % 15 == 0) and (i % 30 > 0) and (i+3 < adj.shape[1]):
            adj[i][i+2], adj[i][i+3], adj[i+1][i+3] = 1, 1, 1
            adj[i+2][i], adj[i+3][i], adj[i+3][i+1] = 1, 1, 1

        # # Create 4-cliques in clique_numbers places (forward)    
        # if i in clique_numbers and (i+3 < adj.shape[1]):
        #     adj[i][i+2], adj[i][i+3], adj[i+1][i+3] = 1
        #     adj[i+2][i], adj[i+3][i], adj[i+3][i+1] = 1

        for j in range(adj.shape[1]):
            if j == i+1: 
                adj[i][j] = 1
                adj[j][i] = 1

    return csr_matrix(adj)

def make_features(adj, num_nodes):
    graph = nx.from_scipy_sparse_array(adj)
    temp_dict = dict(graph.degree)                                             # degree of each node
    temp_clus = dict(nx.clustering(graph))                                     # clutering_coefficient of each node
    temp_closeness = dict(nx.closeness_centrality(graph))                      # closeness centrality for each node
    temp_betw = dict(nx.betweenness_centrality(graph))                         # betweenness_centrality for each node
    clique_dict =  nx.cliques_containing_node(graph)                           # dictionary of largest cliques in each node
    newClique_dict = {key: len(value) for key, value in clique_dict.items()}   # length of the clique in each node
    temp_load = dict(nx.load_centrality(graph))                                # load centrality of each node

    features = np.zeros((num_nodes, 6))
    for i in range(num_nodes): 
        features[i][0] = temp_dict[i]
        features[i][1] = temp_clus[i]
        features[i][2] = temp_closeness[i]
        features[i][3] = temp_betw[i]
        features[i][4] = newClique_dict[i]
        features[i][5] = temp_load[i]

    return csr_matrix(features)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--numNodes')
args = parser.parse_args()

data_name = args.dataset
num_nodes = int(args.numNodes)
print(data_name, num_nodes)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# make adj and features
adj = make_adjacency(num_nodes)
features = make_features(adj, num_nodes)

# store artificial graph info (adj & features) in pickle
# featurePickleFile = 'Raw/{}_features_hop_{}.pickle'.format(data_name, 0)
# adjPickleFile = 'Raw/{}_adj_hop_{}.pickle'.format(data_name, 0)

with open('Raw/{}_features_hop_{}.pickle'.format(data_name, 0), 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Raw/{}_adj_hop_{}.pickle'.format(data_name, 0), 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done storing hopped features and adj in sparse form")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# python makeRandGraph.py --dataset=ArtificialV4 --numNodes=2000
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------