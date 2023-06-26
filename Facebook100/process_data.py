import sys
import multiprocessing
import matplotlib.pyplot as plt
from math import isclose
import networkx as nx
import numpy as np
import pandas as pd
import pickle


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset')

args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
dataset = args.dataset   # 'Bowdoin47', 'Amherst41', 'Bingham82', 'Brown11'
# python process_data.py --dataset=Bingham82

graph_file = 'Raw/' + dataset + '.graphml'
G = nx.read_graphml(graph_file) 
adjDf = nx.to_pandas_adjacency(G)
featureDf = pd.DataFrame.from_dict(G.nodes, orient='index')
print(featureDf)
print(adjDf)

f1 = 'Processed/{}_featuresDf_hop_{}.pickle'.format(dataset, str(0))
a1 = 'Processed/{}_adjDf_hop_{}.pickle'.format(dataset, str(0))
with open(f1, 'wb') as handle: pickle.dump(featureDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(a1, 'wb') as handle: pickle.dump(adjDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Stored in ', f1, 'and ', a1)