# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle


data_name = 'ArtificialV4'
hop = 5

featurePickleFile = 'Processed/{}_features_hop_{}.pickle'.format( data_name, str(hop))
adjPickleFile = 'Processed/{}_adj_hop_{}.pickle'.format(data_name, str(hop))
with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 
with open(featurePickleFile, 'rb') as handle: features = pickle.load(handle)

# adj = adj.todense()
print(adj.sum(axis=None))