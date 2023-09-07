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
        self.n_cluster = n_cluster
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

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, x, adj, finetune=True):
        # augmentations
        x_aug = aug_feature_dropout(x)
        x_shuffle = aug_feature_shuffle(x_aug)

        # discrimination loss
        logit = self.forward(x_aug, x_shuffle, adj, sparse=True)
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)
        loss_disc = self.discrimination_loss(logit, disc_y)

        if finetune:
            # clustering loss
            h = self.embed(x, adj, power=5, sparse=True)
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
            self.no_diag(center_distance, self.cluster_center.shape[0])
            clustering_loss = sample_center_distance.mean() - center_distance.mean()

            # tradeoff
            loss = clustering_loss + self.tradeoff * loss_disc

        else:
            loss = loss_disc
            sample_center_distance = None

        return loss, sample_center_distance

    def clustering(self, x, adj, finetune=True):
        h = self.embed(x, adj, sparse=True)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).cpu().detach().numpy()
        return cluster_results


# ------------------------from dgl------------------------
class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, gnn_encoder, power=10):
        super(Encoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        activation = nn.PReLU(n_hidden) if activation == 'prelu' else activation
        if gnn_encoder == 'gcn':
            self.conv = GCN_dgl(in_feats, n_hidden, n_layers, activation)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=power, cached=True)

    def forward(self, features, g, corrupt=False, batch_train=False):
        if corrupt:
            perm = torch.randperm(features.shape[0])
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(g, features, batch_train)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(g, features, batch_train)
        return features


class GCN_dgl(nn.Module):
    def __init__(self, n_in, n_h, n_layers, activation, bias=True, weight=True):
        super(GCN_dgl, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphConv(n_in, n_h, weight=weight, bias=bias, activation=activation))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_h, n_h, weight=weight, bias=bias, activation=activation))

    def forward(self, g, feat, batch_train=False):
        h = feat.squeeze(0)
        if batch_train:
            for i, layer in enumerate(self.layers):
                h = layer(g[i], h)
        else:
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
        return h


class DinkNet_dgl(nn.Module):
    def __init__(self, g_global, n_in, n_h, n_cluster, tradeoff, encoder_layers, activation, projector_layers=1, dropout_rate=0.2, gnn_encoder='gcn', n_hop=10):
        super(DinkNet_dgl, self).__init__()
        self.g_global = g_global
        self.n_cluster = n_cluster
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_h))
        self.encoder = Encoder(n_in, n_h, encoder_layers, activation, gnn_encoder, n_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(projector_layers):
            self.mlp.append(nn.Linear(n_h, n_h))
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff
        self.dropout_rate = dropout_rate

    def forward(self, x, g, batch_train):
        z_1 = self.encoder(x, g, corrupt=False, batch_train=batch_train)
        z_2 = self.encoder(x, g, corrupt=True, batch_train=batch_train)

        for i, lin in enumerate(self.mlp):
            z_1 = lin(z_1)
            z_2 = lin(z_2)

        logit = torch.cat((z_1.sum(1), z_2.sum(1)), 0)
        return logit

    def embed(self, x, g, power=10, batch_train=False):
        local_h = self.encoder(x, g, corrupt=False, batch_train=batch_train)

        feat = local_h.clone().squeeze(0)

        if batch_train:
            g = dgl.node_subgraph(self.g_global, g[-1].dstdata["_ID"].to(self.g_global.device)).to(feat.device)

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

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, x, g, batch_train=False, finetune=True):
        # augmentations
        x_aug = aug_feature_dropout(x, drop_rate=self.dropout_rate).squeeze(0)

        logit = self.forward(x_aug, g, batch_train=batch_train)

        # label of discriminative task
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)

        # discrimination loss
        loss_disc = self.discrimination_loss(logit, disc_y)

        if finetune:
            # clustering loss
            h = self.embed(x, g, power=10, batch_train=batch_train)
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
            self.no_diag(center_distance, self.cluster_center.shape[0])
            clustering_loss = sample_center_distance.mean() - center_distance.mean()

            # tradeoff
            loss = clustering_loss + self.tradeoff * loss_disc

        else:
            loss = loss_disc
            sample_center_distance = None

        return loss, sample_center_distance

    def clustering(self, x, adj, batch_train=False, finetune=True):
        h = self.embed(x, adj, power=10, batch_train=batch_train)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).cpu().detach().numpy()
        return cluster_results
