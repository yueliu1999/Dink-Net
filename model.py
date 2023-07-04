import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_in, n_h, num_cluster, tradeoff=1e-10, activation="prelu"):
        super(DinkNet, self).__init__()
        self.cluster_center = torch.nn.Parameter(torch.Tensor(num_cluster, n_h))
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

    def cal_loss(self, x, x_aug, x_shuffle, adj, disc_y):
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
