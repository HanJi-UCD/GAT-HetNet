import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    
    
class GCN_node(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_node, self).__init__()
        self.conv1 = GCNConv(input_dim, int(hidden_dim/2))
        self.bn1 = nn.BatchNorm1d(int(hidden_dim / 2))
        self.conv2 = GCNConv(int(hidden_dim/2), output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    
class GAT_node(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2, dropout=0.5):
        super(GAT_node, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    
class GAT_L3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2, dropout=0.5):
        super(GAT_L3, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        return x
    
class GraphSAGE_node(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE_node, self).__init__()
        self.conv1 = SAGEConv(input_dim, int(hidden_dim/2))
        self.bn1 = nn.BatchNorm1d(int(hidden_dim/2))
        self.conv2 = SAGEConv(int(hidden_dim/2), output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x
     
class TCNN(nn.Module):
    def __init__(self, net='resnet18', UE_num = 20, in_dim = 18, out_dim = 3, dropout_prob=0.5):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(UE_num*in_dim, 128, bias=True), #1
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(128, 64, bias=True), #2
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(64, 16, bias=True), #3
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(inplace=True), # third layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(16, 8, bias=True), 
                                        nn.BatchNorm1d(8),
                                        nn.ReLU(inplace=True), #
                                        )
        self.target = nn.Sequential(nn.Linear(in_dim, 16, bias=True), #1
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(16, 8, bias=True), #3
                                        nn.BatchNorm1d(8),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(8, 4, bias=True), #3
                                        nn.BatchNorm1d(4),
                                        nn.ReLU(inplace=True), # third layer
                                        )
        self.combiner = nn.Sequential(nn.Linear(12, out_dim, bias=False),
                                        nn.BatchNorm1d(out_dim),
                                        nn.Sigmoid()
                                        )

    def forward(self, tar, cond):
        tar = self.target(tar)        
        cond = self.condition(cond)
        x = torch.cat((tar, cond), dim=1)
        x = self.combiner(x)
        return x
    
    
class DNN(nn.Module):
    def __init__(self, net='resnet18', UE_num = 20, in_dim = 18, out_dim = 3, dropout_prob=0.5):
        super(DNN, self).__init__()
        self.FC = nn.Sequential(nn.Linear(UE_num*in_dim, 128, bias=True), #1
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(128, 64, bias=True), #2
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(64, 64, bias=True), #3
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # third layer
                                        nn.Dropout(dropout_prob),  # dropout
                                        
                                        nn.Linear(64, out_dim*UE_num, bias=True), 
                                        nn.BatchNorm1d(out_dim*UE_num),
                                        nn.Sigmoid(), #
                                        )

    def forward(self, ipt):
        output = self.FC(ipt)
        return output
    
    
    
    
    
################## need revise ##################
class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None
        
        self.fc3_6_class = nn.Linear(output_dim, 6)
        self.fc3_5_class = nn.Linear(output_dim, 5)

    def forward(self, data, edge_SF_num):
        x, edge_index = data.x, data.edge_index
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]
            
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")
            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")
        
        y = self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x)
        x_SF = self.fc3_6_class(y)
        x_Ptx = self.fc3_5_class(y)
        
        return x_SF, x_Ptx
    
    
def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")
        
def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)
    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj










