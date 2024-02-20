import torch
from torch.nn import Linear
import torch.nn.functional as F
from pump import *
from torch_geometric.utils import spmm
class EIGENX(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_eigen,num_nodes,out_channels,drop_out = 0.5):
        super(EIGENX, self).__init__()
        self.MLP = Linear(num_nodes, num_eigen)
        self.classify1 = Linear(num_eigen, out_channels)
        #Extra conv
        self.mlpx = Linear(in_channels, hidden_channels)
        # Aux
        #self.mlpW = Linear(hidden_channels + num_eigen, hidden_channels)
        self.mlpFinal = Linear(hidden_channels + num_eigen, out_channels)
        self.drop_out = drop_out
        self.num_nodes = num_nodes
    def forward(self, data):
        row, col = data.edge_index
        indices = torch.stack([col, row], dim=0)
        values = torch.ones(data.edge_index.size(1))
        adj = torch.sparse_coo_tensor(indices, values, size=(self.num_nodes, self.num_nodes), dtype=data.x.dtype, device=data.x.device)
        # MLP(A)
        xA = spmm(adj, self.MLP.weight.t(), reduce='sum')
        xA += self.MLP.bias
        # MLP(X)
        xX = self.mlpx(data.x)
        # Pump
        xA, pump_loss, ortho_loss= pump(xA,adj)
        xA = xA.squeeze(0)
        z = torch.cat([xX, xA], dim=1)
        z = F.dropout(z, p=self.drop_out, training=self.training)
        z = self.mlpFinal(z).log_softmax(dim=-1)
        return z, (pump_loss + ortho_loss)
