import pandas as pd
import glob
import os
import networkx as nx
import pickle
import numpy as np
from scipy import sparse

edgesFile = 'Raw/edges.csv'
nodesFile = 'Raw/nodes.csv'

edgesDf = pd.read_csv(edgesFile, sep=',')
edgesDf = edgesDf[["source", "sink"]]
print('edgesDf shape', edgesDf.shape)

nodesDf = pd.read_csv(nodesFile, sep=',')
print('nodesDf shape', nodesDf.shape)

# number of NaN and unique per column
# print(nodesDf.isnull().sum(axis = 0))
# print(nodesDf.nunique())

# remove nodes with empty pval
nodesPvalDf = nodesDf[nodesDf['pval'].notna()]
print('nodesPvalDf shape', nodesPvalDf.shape)

# number of NaN and unique values per column
# print(nodesPvalDf.isnull().sum(axis = 0))
# print(nodesPvalDf.nunique())

nodesSelectedCols = nodesPvalDf[['node_id', 'chromosome', 'pval']].dropna()
print('nodesSelectedCols shape', nodesSelectedCols.shape)

# one hot encode chromosome
oneHot = pd.get_dummies(nodesSelectedCols['chromosome'])
nodesOneHot = nodesSelectedCols.join(oneHot)
nodesOneHot = nodesOneHot.drop('chromosome', axis = 1)

print(nodesOneHot)

print('Total unique values for rs_id=', len(nodesSelectedCols.rs_id.unique()))

# sort by node_id
nodesSelectedCols = nodesSelectedCols.sort_values(by=['node_id'])

nodeList = nodesSelectedCols["node_id"].values.tolist()
print('nodeList len', len(nodeList))

# drop node_id column
nodesSelectedCols = nodesSelectedCols.drop('node_id', axis=1).reset_index(drop=True)
print(nodesSelectedCols)

# filter edges that are not in selected nodes
edgesDfFiltered = edgesDf[edgesDf['source'].isin(nodeList)]
edgesDfFiltered = edgesDfFiltered[edgesDfFiltered['sink'].isin(nodeList)]
print('edgesDfFiltered shape', edgesDfFiltered.shape)

# build adj matrix
numNodes = len(nodeList)
adj = np.zeros((numNodes, numNodes))

for index, row in edgesDfFiltered.iterrows():
    u = nodeList.index(row['source'])
    v = nodeList.index(row['sink'])
    adj[u][v] = 1
    adj[v][u] = 1

adj = sparse.csr_matrix(adj)
print('adj shape', adj.shape)

# f1 = 'Processed/features_human.pickle'
# a1 = 'Processed/adj_human.pickle'

# with open(f1, 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(a1, 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print('Stored in', f1, 'and', a1)

