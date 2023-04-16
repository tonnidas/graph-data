
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import defaultdict
from scipy import sparse


def getNHopNeighbors(node, hop, adjList): # It is simply a bfs till nhop, not on whole graph
    neighborsTillHop, n_neighbors = set(), {node}

    for i in range(hop):
        temp = set()

        for curNode in n_neighbors:
            if curNode in adjList:
                temp = temp.union(set(adjList[curNode]))
        
        neighborsTillHop = neighborsTillHop.union(temp)
        n_neighbors = temp
    
    return neighborsTillHop

# converts from adjacency matrix to adjacency list
def convert(numNodes, adj):
    adj = adj.todense()
    adjList = defaultdict(list) # Type: Default value is empty list
    for i in range(numNodes):
        for j in range(numNodes):
                if adj[i,j] > 0:
                    adjList[i].append(j)
    return adjList

def addHopFeatures(features, adj, hop_count):
    print('features_n_hop start')

    numNodes = features.shape[0]

    adjList = convert(numNodes, adj)

    n_hop_neighbors = hop_count

    Vertices_attributes_oneHot = pd.DataFrame.sparse.from_spmatrix(features)

    all_nodes_distribution = np.zeros((numNodes, len(Vertices_attributes_oneHot.columns)))

    for eachNode in range(numNodes):
        Immediate_friends_Nodes = getNHopNeighbors(eachNode, n_hop_neighbors, adjList) # gets a list of adjacent nodes till n hop
        # Vertices_attributes_sum = Vertices_attributes_oneHot.iloc[list(Immediate_friends_Nodes)].sum()
        Vertices_attributes_sum = Vertices_attributes_oneHot.iloc[list(Immediate_friends_Nodes)].max()
        Vertices_attributes_sum = Vertices_attributes_sum.to_numpy()
        # Vertices_attributes_sum[Vertices_attributes_sum > 0] = 1 # replace non-zero with 1
        all_nodes_distribution[eachNode] = Vertices_attributes_sum

    features_n_hop = sparse.csr_matrix(all_nodes_distribution) # convert to sparse matrix

    # with open('features_n_hop.pickle', 'wb') as handle: pickle.dump(features_n_hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('features_n_hop done')

    return features_n_hop

def addHopAdjacency(adj, hop_count):
    print('adjacency_n_hop start')
    print("---------------------")

    numNodes = adj.shape[0]           # 6271 for mmu
    print('numNodes:', numNodes)

    adjList = convert(numNodes, adj)

    n_hop_neighbors = hop_count

    nHopAdj = np.zeros((numNodes, numNodes), dtype=int)

    for eachNode in range(numNodes):
        Immediate_friends_Nodes = getNHopNeighbors(eachNode, n_hop_neighbors, adjList) # gets a list of adjacent nodes till n hop

        for friends_Node in Immediate_friends_Nodes:
            nHopAdj[eachNode][friends_Node] = 1

    nHopAdj = sparse.csr_matrix(nHopAdj) # convert to sparse matrix

    # with open('KEGG_pickles/adj_n_hop.pickle', 'wb') as handle: pickle.dump(nHopAdj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('adj_n_hop done')

    return nHopAdj
    