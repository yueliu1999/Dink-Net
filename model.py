from utils import *
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# ------------------------from scratch------------------------
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.FloatTensor(out_ft))

        # init parameters
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        self.bias.data.fill_(0.0)

    def forward(self, feat, adj, sparse=False):
        h = self.fc(feat)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(h, 0)), 0)
        else:
            out = torch.bmm(adj, h)
        out += self.bias
        return self.act(out)


class DinkNet(nn.Module):
    def __init__(self, n_in, n_h, n_cluster, tradeoff=1e-10, activation="prelu"):
        super(DinkNet, self).__init__()
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_h))
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff

    def forward(self, x_1, x_2, adj, sparse):
        h_1 = self.gcn(x_1, adj, sparse)
        h_2 = self.gcn(x_2, adj, sparse)
        z_1 = ((self.lin(h_1.squeeze(0))).sum(1))
        z_2 = ((self.lin(h_2.squeeze(0))).sum(1))
        logit = torch.cat((z_1, z_2), 0)
        return logit

    def embed(self, x, adj, power=5, sparse=True):
        local_h = self.gcn(x, adj, sparse)
        global_h = local_h.clone().squeeze(0)
        for i in range(power):
            global_h = adj @ global_h
        global_h = global_h.unsqueeze(0)
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])
        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)
        return h

    def cal_loss(self, x, adj, disc_y):
        # augmentations
        x_aug = aug_feature_dropout(x)
        x_shuffle = aug_feature_shuffle(x_aug)

        # discrimination loss
        logit = self.forward(x_aug, x_shuffle, adj, sparse=True)
        loss_disc = self.discrimination_loss(logit, disc_y)

        # clustering loss
        h = self.embed(x, adj, power=5, sparse=True)
        sample_center_distance = (h.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        center_distance = (self.cluster_center.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        center_distance.flatten()[:-1].view(self.cluster_center.shape[0] - 1,
                                            self.cluster_center.shape[0] + 1)[:, 1:].flatten()
        clustering_loss = sample_center_distance.mean() - center_distance.mean()

        # tradeoff
        loss = clustering_loss + self.tradeoff * loss_disc
        return loss, sample_center_distance

    def clustering(self, x, adj):
        h = self.embed(x, adj, sparse=True)
        sample_center_distance = (h.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        cluster_results = torch.argmin(sample_center_distance, dim=-1)
        return cluster_results.cpu().detach().numpy()


# ------------------------from dgl------------------------
class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        activation = nn.PReLU(n_hidden) if activation == 'prelu' else activation
        if gnn_encoder == 'gcn':
            self.conv = GCN_dgl(g, in_feats, n_hidden, n_layers, activation)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(features)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, features)
        return features


class GCN_dgl(nn.Module):
    def __init__(self, g, n_in, n_h, n_layers, activation, bias=True, weight=True):
        super(GCN_dgl, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphConv(n_in, n_h, weight=weight, bias=bias, activation=activation))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_h, n_h, weight=weight, bias=bias, activation=activation))

    def forward(self, feat):
        h = feat
        h = h.squeeze(0)
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h


class DinkNet_dgl(nn.Module):
    def __init__(self, g, n_in, n_h, n_cluster, tradeoff, n_layers, activation, proj_layers=1, gnn_encoder='gcn', n_hop=10):
        super(DinkNet_dgl, self).__init__()
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_h))
        self.encoder = Encoder(g, n_in, n_h, n_layers, activation, gnn_encoder, n_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_h, n_h))
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff

    def forward(self, x):
        z_1 = self.encoder(x, corrupt=False)
        z_2 = self.encoder(x, corrupt=True)

        for i, lin in enumerate(self.mlp):
            z_1 = lin(z_1)
            z_2 = lin(z_2)

        logit = torch.cat((z_1.sum(1), z_2.sum(1)), 0)
        return logit

    def embed(self, x, g, power=10):
        local_h = self.encoder(x, corrupt=False)

        feat = local_h.clone().squeeze(0)

        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).unsqueeze(1).to(local_h.device)
        for i in range(power):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'), fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        global_h = feat.unsqueeze(0)
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])

        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)

        return h

    def cal_loss(self, x, g, disc_y):
        # augmentations
        x_aug = aug_feature_dropout(x).squeeze(0)

        # discrimination loss
        logit = self.forward(x_aug)
        loss_disc = self.discrimination_loss(logit, disc_y)

        # clustering loss
        h = self.embed(x, g, power=10)
        sample_center_distance = (h.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        center_distance = (self.cluster_center.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        center_distance.flatten()[:-1].view(self.cluster_center.shape[0] - 1,
                                            self.cluster_center.shape[0] + 1)[:, 1:].flatten()
        clustering_loss = sample_center_distance.mean() - center_distance.mean()

        # tradeoff
        loss = clustering_loss + self.tradeoff * loss_disc

        return loss, sample_center_distance

    def clustering(self, x, adj):
        h = self.embed(x, adj, power=10)
        sample_center_distance = (h.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(-1)
        cluster_results = torch.argmin(sample_center_distance, dim=-1)
        return cluster_results.cpu().detach().numpy()
