import pandas as pd
import glob
import os
import networkx as nx
import pickle

# sort datasets by number of rows
def sort_datasets():
    files = os.path.join("Raw/kegg_gene_network_hsa", "*.tsv")
    files = glob.glob(files)

    datasets = []

    for f in files:
        df = pd.read_csv(f, sep='\t')
        datasetName = os.path.splitext(os.path.basename(f))[0]
        datasets.append((df.shape[0], datasetName))

    datasets.sort(reverse=True)
    return datasets

# ================================================================================================================================================================

def store_adj(rawFile, outputFile):
    df = pd.read_csv(rawFile, sep='\t')
    G = nx.from_pandas_edgelist(df, 'entry1', 'entry2')
    
    print('nodes:', len(G.nodes), "edges:", len(G.edges))

    A = nx.adjacency_matrix(G)
    with open(outputFile, 'wb') as handle:
        pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ================================================================================================================================================================

datasets = sort_datasets()

# process 10 largest datset
for i in range(5):
    dataset = datasets[i][1]
    rawFile = 'Raw/kegg_gene_network_hsa/{}.tsv'.format(dataset)
    rawFileUnque = 'Raw/kegg_gene_network_hsa_unique/{}.tsv'.format(dataset)
    outputFile = 'Processed/{}_adj.pickle'.format(dataset)
    outputFileUnque = 'Processed/{}_unique_adj.pickle'.format(dataset)

    print('Processing dataset: {}'.format(dataset))
    store_adj(rawFile, outputFile)

    print('Processing dataset: {}_unique'.format(dataset))
    store_adj(rawFileUnque, outputFileUnque)
