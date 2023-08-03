import dgl
import sys
import copy
import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from munkres import Munkres
from collections import Counter
from ogb.nodeproppred import DglNodePropPredDataset
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


def load_data(args):
    """
    Load data for cora and citeseer datasets.
    args: 
        args: parameters
    returns:
        features: node attributes
        sp_adj: sparse normalized adjacency matrix
        labels: node labels
        n: number of nodes
        k: number of clusters
        d: dimension number of node attributes
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(args.dataset_dir, args.dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    filename = "{}/ind.{}.test.index".format(args.dataset_dir, args.dataset)
    test_idx_reorder = []
    for line in open(filename):
        test_idx_reorder.append(int(line.strip()))

    test_idx_range = np.sort(test_idx_reorder)

    if args.dataset == 'citeseer':
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


def load_amazon_photo():
    """
    Load data for amazon photo dataset.
    args:
    returns:
        features: node attributes
        g: dgl graph
        labels: node labels
        n: number of nodes
        k: number of clusters
        d: dimension number of node attributes
    """

    adj, features, labels, train_mask, val_mask, test_mask = load_pitfall_dataset("photo")
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    features = preprocess_features(features)

    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(features)
    g.ndata['label'] = torch.LongTensor(labels)

    g = g.remove_self_loop().add_self_loop()

    n = g.ndata['feat'].shape[0]
    k = labels.shape[-1]
    d = features.shape[-1]

    labels = labels.argmax(1)
    features = g.ndata['feat']

    return features, g, labels, n, k, d


def load_data_ogb(args):
    """
    Load data for ogbn-arxiv dataset.
    args:
        args: parameters
    returns:
        features: node attributes
        g: dgl graph
        labels: node labels
        n: number of nodes
        k: number of clusters
        d: dimension number of node attributes
        train_loader: dataloader of the graph
    """

    data = DglNodePropPredDataset(name=args.dataset.replace('_', '-'), root=args.dataset_dir)
    g, labels = data[0]

    feat = g.ndata["feat"]
    g = dgl.to_bidirected(g)
    g.ndata["feat"] = feat

    g = g.remove_self_loop().add_self_loop()
    g.create_formats_()

    features = g.ndata['feat']
    labels = labels.reshape(-1, ).numpy()

    n = features.shape[0]
    k = labels.max() + 1
    d = features.shape[-1]

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.encoder_layer)

    train_loader = dgl.dataloading.DataLoader(
        g, torch.tensor(range(n)), sampler,
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    test_loader = dgl.dataloading.DataLoader(
        g, torch.tensor(range(n)), sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    return features, g, labels, n, k, d, train_loader, test_loader


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


class SparseGraph:
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):

        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        return self.adj_matrix.shape[0]

    def num_edges(self):
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        return self.adj_matrix[idx].indices

    def is_directed(self):
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def standardize(self):
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(data_path):
    if not str(data_path).endswith('.npz'):
        data_path = data_path.joinpath('.npz')

    with np.load(data_path, allow_pickle=True) as loader:
        loader = dict(loader)

        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None
        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def eliminate_self_loops(G):
    def remove_self_loop(A):
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()
        A.eliminate_zeros()
        return A

    G.adj_matrix = remove_self_loop(G.adj_matrix)
    return G


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def to_binary_bag_of_words(features):
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def largest_connected_components(sparse_graph, n_components=1):
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def load_pitfall_dataset(dataset_str,
                         standardize_graph=True,
                         train_ratio=0.1,
                         val_ratio=0.1):
    file_map = {'computer':'amazon_electronics_computers.npz',
                'photo':'amazon_electronics_photo.npz',
                'phy': 'ms_academic_phy.npz',
                'cs': 'ms_academic_cs.npz'}
    file_name = file_map[dataset_str]
    data_path = './data/' + file_name
    dataset_graph = load_npz_to_sparse_graph(data_path)

    if standardize_graph:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    adj, features, labels = dataset_graph.unpack()
    labels = binarize_labels(labels)

    if not is_binary_bag_of_words(features):
        features = to_binary_bag_of_words(features)

    assert (adj != adj.T).nnz == 0
    assert is_binary_bag_of_words(features), f"Non-binary node_features entry!"

    idx_train, idx_val, idx_test = get_train_val_test_split(labels,train_examples_per_class = 30, val_examples_per_class=30)

    return adj, features, labels, idx_train, idx_val, idx_test


def get_train_val_test_split(labels,
                             train_examples_per_class=30, val_examples_per_class=30,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices
