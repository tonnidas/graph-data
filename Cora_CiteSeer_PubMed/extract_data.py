import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# Params:
# dataset = name of the dataset
# Return values:
# adj = bidirection adj metrix of size (num_nodes, num_nodes), for cora (2708, 2708)
# features = attributes of all nodes of size (num_nodes, num_attributes), for cora (2708, 1433)
# y_test = one-hot labels of test data of size (num_nodes, num_labels) where last num_test_data rows are non-zero, for cora size is (2708, 7) and last 1000 rows are non-zero
# tx = attributes of test data of size (num_test_data, num_attributes), for cora (1000, 1433)
# ty = one-hot labels of test_data of size (num_test_data, num_labels), for cora size is (1000, 7)
# test_maks = an boolean array of size num_nodes where last num_test_data values are true, for cora size is 2708 and last 1000 values are true
# true_labels = reverse one-hot of all nodes, an array of size num_nodes, for cora size is 2708
def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pickle.load(open("Raw/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("Raw/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    # To see how the dataset looks like
    # print("len(x)", x.shape[0], "col(x)", x.shape[1], "x[0]\n", x[0], "\nx[1707]", x[-1])
    # print("len(tx)", tx.shape[0], "col(tx)", tx.shape[1], "tx[0]\n", tx[0], "\ntx[99]\n", tx[99])
    # print("len(ty)", len(ty), "col(ty)", len(ty[0]), "ty[0]\n", ty[0], "\nty[99]\n", ty[99])
    # print("len(allx)", allx.shape[0], "col(allx)", allx.shape[1], "allx[0]\n", allx[0], "\nallx[99]\n", allx[-1])
    # print("len(graph)", len(graph), "len(test_idx_reorder)", len(test_idx_reorder))

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # Labels are cluster identity of each feature vector or row
    labels = np.vstack((ally, ty)) # labels = [[0 0 0 0 1 0 0],[0 0 0 1 0 0 0], ..., [0 0 0 0 1 0 0]], size of labels = 2708
    labels[test_idx_reorder, :] = labels[test_idx_range, :] # Keeping labels array in ascending order

    idx_test = test_idx_range.tolist() # List of all test nodes (1708, ... , 2707)
    idx_train = range(len(y)) # List of all train nodes (0, ... , 139)
    idx_val = range(len(y), len(y) + 500) # List of 500 nodes (140, ... , 639)

    train_mask = sample_mask(idx_train, labels.shape[0]) # List of size 2708, only 0 to 139 rows are True
    val_mask = sample_mask(idx_val, labels.shape[0]) # List of size 2708, only 140 to 639 rows are True
    test_mask = sample_mask(idx_test, labels.shape[0]) # List of size 2708, only 1708 to 2707 are True

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_test, tx, ty, test_mask, np.argmax(labels,1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
  
args = parser.parse_args()
print('Arguments:', args)

data_name = args.dataset   # 'cora' or 'citeseer' or 'pubmed'
# Command to run
# python extract_data.py --dataset=cora
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load data
adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)
f1 = 'Processed/' + data_name + '_features_hop_' + str(0) + '.pickle'
a1 = 'Processed/' + data_name + '_adj_hop_' + str(0) + '.pickle'
print('Stored in ', f1, 'and ', a1)
with open(f1, 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(a1, 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)