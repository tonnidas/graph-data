# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import os

from hopInfo import addHopFeatures, addHopAdjacency

def prepare_hopped_graph(folder_name, data_name, hop):

    # Load the 0-hopped (original) data from stellargraph
    featurePickleFile = '{}/Processed/{}_features_hop_{}.pickle'.format(folder_name, data_name, 0)
    adjPickleFile = '{}/Processed/{}_adj_hop_{}.pickle'.format(folder_name, data_name, 0)
    with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 
    with open(featurePickleFile, 'rb') as handle: features = pickle.load(handle)

    if hop != 0:
        # Store hopped info in pickle
        features = addHopFeatures(features, adj, hop)
        adj = addHopAdjacency(adj, hop + 1)

    print(type(adj))
    print(type(features))

    f1 = '{}/Processed/{}_features_hop_{}.pickle'.format(folder_name, data_name, str(hop))
    a1 = '{}/Processed/{}_adj_hop_{}.pickle'.format(folder_name, data_name, str(hop))
    with open(f1, 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(a1, 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done storing hopped features and adj in sparse form")
# =======================================================================


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--folderName')
parser.add_argument('--dataset')
parser.add_argument('--hop')

args = parser.parse_args()
print('Arguments:', args)

folder_name = args.folderName
data_name = args.dataset       # 'CiteSeer' or 'Cora' or 'PubMed'
hop_count = int(args.hop)
print(data_name)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# python prepare_hopInfo.py --folderName=Cora_CiteSeer_PubMed --dataset=cora --hop=1
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


prepare_hopped_graph(folder_name, data_name, hop_count)
print('Done preparing and storing ' + str(hop_count) + ' hopped features and adjacency for = ' + data_name)
