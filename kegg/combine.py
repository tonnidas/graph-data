# combine hsa and mmu dataset

import pandas as pd
import glob
import os
import networkx as nx
import pickle

dataset = '04010'
hsaRawFile = 'Raw/kegg_gene_network_hsa/hsa{}.tsv'.format(dataset)
mmuRawFile = 'Raw/kegg_gene_network_mmu/mmu{}.tsv'.format(dataset)
outputFile = 'Processed/hsa_mmu_{}_adj.pickle'.format(dataset)

hsaDf = pd.read_csv(hsaRawFile, sep='\t')
mmuDf = pd.read_csv(mmuRawFile, sep='\t')
df = pd.concat([hsaDf, mmuDf], ignore_index=True)

G = nx.from_pandas_edgelist(df, 'entry1', 'entry2')
    
print('nodes:', len(G.nodes), "edges:", len(G.edges))

A = nx.adjacency_matrix(G)
with open(outputFile, 'wb') as handle:
    pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
