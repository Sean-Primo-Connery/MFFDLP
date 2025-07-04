import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GNN(nn.Module):
    def __init__(self, _hidden_dims, _dropout=0.2):
        super(GNN, self).__init__()
        self.gnn1 = GATv2Conv(_hidden_dims[0], _hidden_dims[1], heads=4, concat=False, residual=True, dropout=_dropout)
        self.gnn2 = GATv2Conv(_hidden_dims[1], _hidden_dims[2], heads=4, concat=False, residual=True, dropout=_dropout)

        self.bn1 = nn.BatchNorm1d(_hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(_hidden_dims[2])

        self.act1 = nn.PReLU(_hidden_dims[1])
        self.act2 = nn.PReLU(_hidden_dims[2])

        self.dropout = nn.Dropout(_dropout)

    def forward(self, x, edge_index):

        x1 = self.gnn1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.dropout(x1)

        x2 = self.gnn2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.dropout(x2)

        x_all = torch.cat([x1, x2], dim=-1)

        return x_all


class MLP(nn.Module):
    def __init__(self, in_dim=23, hid1=64, hid2=16, out_dim=2, p_drop=0.):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)

        self.fc1 = nn.Linear(in_dim, hid1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p_drop)
        self.norm1 = nn.BatchNorm1d(hid1)

        self.fc2 = nn.Linear(hid1, hid2)
        self.norm2 = nn.BatchNorm1d(hid2)

        self.fc3 = nn.Linear(hid2, out_dim)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)

        return self.fc3(x)


class RNN(nn.Module):
    def __init__(self, _input_size, _hidden_size, _num_layers, _dropout=0.2):
        super(RNN, self).__init__()
        self.num_layers = _num_layers
        self.rnn = nn.GRU(_input_size, _hidden_size, self.num_layers, bidirectional=True, batch_first=True, dropout=_dropout) if self.num_layers > 1 else nn.GRU(_input_size, _hidden_size, self.num_layers, bidirectional=True, batch_first=True)

        self.agg = nn.Linear(5, 1)

        self.dropout = nn.Dropout(_dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, hn = self.rnn(x)
        output = self.dropout(output)

        x_feature = self.agg(output.permute(0, 2, 1)).flatten(1)
        return x_feature


class Model_CN(nn.Module):
    def __init__(self):
        super(Model_CN, self).__init__()
        self.gnn_all = GNN([128, 16, 16], _dropout=0.5)
        self.mlp = MLP(in_dim=86, hid1=32, hid2=8, out_dim=2, p_drop=0.5)

    def forward(self, x_all_year, label_edge_index, edge_index_all_year, label_edge_attr_all_year):
        x_all = self.gnn_all(x_all_year, edge_index_all_year)

        src_feature_all = x_all[label_edge_index[0]]
        dst_feature_all = x_all[label_edge_index[1]]

        node2edge_feature_all = torch.cat([src_feature_all, dst_feature_all], dim=1)
        label_edge_feature_all = torch.cat([node2edge_feature_all, label_edge_attr_all_year], dim=1)

        return label_edge_feature_all, self.mlp(label_edge_feature_all)


class Model_TN(nn.Module):
    def __init__(self):
        super(Model_TN, self).__init__()
        self.gnn_per = GNN([128, 16, 16], _dropout=0.5)
        self.rnn = RNN(86, 32, 2, _dropout=0.5)
        self.mlp = MLP(in_dim=64, hid1=32, hid2=8, out_dim=2, p_drop=0.5)
        self.ln = nn.BatchNorm1d(86)

    def forward(self, x_per_year, label_edge_index, edge_index_every_year, label_edge_feature_every_year):
        x_lstm = []
        for (x, edge_index, label_edge_feature) in zip(x_per_year, edge_index_every_year, label_edge_feature_every_year):
            x_per = self.gnn_per(x, edge_index)
            src_feature = x_per[label_edge_index[0]]
            dst_feature = x_per[label_edge_index[1]]
            node2edge_feature = torch.cat([src_feature, dst_feature], dim=1)
            x_lstm.append(self.ln(torch.cat([node2edge_feature, label_edge_feature], dim=1)))

        x_lstm = torch.stack(x_lstm, dim=0)
        x_lstm = self.rnn(x_lstm)
        return x_lstm, self.mlp(x_lstm)


class MFFDLP(nn.Module):
    def __init__(self):
        super(MFFDLP, self).__init__()
        self.model_cn = Model_CN()
        self.model_tn = Model_TN()
        self.final_mlp = MLP(in_dim=4, hid1=32, hid2=8, out_dim=2, p_drop=0.5)
        # self.bn_feature = nn.BatchNorm1d(150)
        self.act = nn.Softmax(dim=-1)

    def forward_a(self, x_all_year, label_edge_index, edge_index_all_year,  label_edge_attr_all_year):
        feature_a, out_a = self.model_cn(x_all_year, label_edge_index, edge_index_all_year, label_edge_attr_all_year)
        return feature_a, out_a

    def forward_b(self, x_per_year, label_edge_index, edge_index_every_year, label_edge_feature_every_year):
        feature_b, out_b = self.model_tn(x_per_year, label_edge_index, edge_index_every_year, label_edge_feature_every_year)
        return feature_b, out_b

    def forward(self, x_per_year, x_all_year, label_edge_index, edge_index_every_year, label_edge_feature_every_year, edge_index_all_year,  label_edge_attr_all_year):
        feature_a, out_a = self.forward_a(x_all_year, label_edge_index, edge_index_all_year, label_edge_attr_all_year)
        feature_b, out_b = self.forward_b(x_per_year, label_edge_index, edge_index_every_year, label_edge_feature_every_year)
        # feature_ab = self.bn_feature(torch.cat([feature_a, feature_b], dim=1))
        out_a = self.act(out_a)
        out_b = self.act(out_b)
        all_feature = torch.cat([out_a, out_b], dim=1)

        return self.final_mlp(all_feature)
