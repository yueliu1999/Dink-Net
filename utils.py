import sys
import copy
import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def evaluation(y_true, y_pred):
    """
    evaluate the clustering performance.
    args:
        y_true: ground truth
        y_pred: prediction
    returns:
        acc: accuracy
        nmi: normalized mutual information
        ari: adjust rand index
        f1: f1 score
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    acc, nmi, ari, f1 = map(lambda x: round(x * 100, 2), [acc, nmi, ari, f1])
    return acc, nmi, ari, f1


def setup_seed(seed):
    """
    fix the random seed.
    args:
        seed: the random seed
    returns:
        none
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = copy.deepcopy(input_feat).squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_feat = aug_input_feat.unsqueeze(0)

    return aug_input_feat


def aug_feature_shuffle(input_feat):
    """
    shuffle the features for fake samples.
    args:
        input_feat: input features
    returns:
        aug_input_feat: augmented features
    """
    fake_input_feat = input_feat[:, np.random.permutation(input_feat.shape[1]), :]
    return fake_input_feat


def load_data(dataset_str):
    """
    Load data for cora and citeseer datasets.
    args: 
        dataset_str: dataset name
    returns:
        sp_adj: sparse normalized adjacency matrix
        features: node attributes
        labels: node labels
        n: number of nodes
        k: number of clusters
        d: dimension number of node attributes
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    filename = "data/ind.{}.test.index".format(dataset_str)
    test_idx_reorder = []
    for line in open(filename):
        test_idx_reorder.append(int(line.strip()))

    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = preprocess_features(features)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = torch.FloatTensor(features).unsqueeze(0)

    n = features.shape[1]
    k = labels.shape[-1]
    d = features.shape[-1]

    labels = labels.argmax(1)

    return features, sp_adj, labels, n, k, d


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """
    convert sparse matrix to tuple representation.
    args: 
        sparse_mx: sparse matirx
        insert_batch: set insert_batch=True if you want to insert a batch dimension.
    returns:
        sparse_mx: tuple representation
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adj):
    """
    symmetrically normalize adjacency matrix.
    args:
        adj: original adjacency matrix
    returns:
        norm_adj: normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return norm_adj


def preprocess_features(features):
    """
    row-normalize node attributes
    args:
        features: input node attributes
    returns:
        normalized node attributes
    """
    rowsum = np.array(features.sum(1))
    rowsum[rowsum==0] = -np.inf
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = sp.diags(r_inv)
    norm_features = r_mat_inv.dot(features).todense()
    return norm_features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    convert a scipy sparse matrix to a torch sparse tensor.
    args:
        sparse_mx: sparse matrix
    returns:
        sparse_tensor: sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return sparse_tensor
