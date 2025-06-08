import torch
from torch import nn
from torch.nn import Linear, ReLU, SiLU, Sequential
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_scatter import scatter
import math

class EGNNLayer(MessagePassing):
    """E(n) Equivariant GNN Layer

    Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.
    """
    def __init__(self, emb_dim, proj_dim=10, activation="relu", norm="layer", aggr="add"):
        """
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + proj_dim**2 + 1, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        self.mlp_pos = Sequential(
            #Linear(emb_dim, emb_dim), self.norm(emb_dim), self.activation, 
           Linear(emb_dim, 1)
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

        # RBF
        self.rbf = NewExpNormalSmearing(num_rbf=proj_dim, rbound_upper=1., rbound_lower=0.0)
        self.sinc = SincRadialBasis(num_rbf=proj_dim, rbound_upper=.1, rbf_trainable=False)

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        projectors = pos_i * pos_j
        #print(projectors[:3])
        projectors = self.sinc(projectors.unsqueeze(-1)).flatten(start_dim=-2, end_dim=-1)
        #print(projectors[:3])
        msg = torch.cat([h_i, h_j, projectors, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        pos_msg = self.mlp_pos(msg)
        avg_pos_msg = torch.mean(pos_msg, 0, True).expand_as(pos_msg)
        #print(torch.exp((pos_msg - avg_pos_msg)/avg_pos_msg)[:3])
        pos_diff = (pos_i - pos_j) * torch.exp((pos_msg - avg_pos_msg)/torch.sqrt(avg_pos_msg))
        #print(pos_diff[:3])
        #import sys; sys.exit()
        # NOTE: some papers divide pos_diff by (dists + 1) to stabilise model.
        # NOTE: lucidrains clamps pos_diff between some [-n, +n], also for stability.
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="sum")
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        """Vanilla Message Passing GNN layer
        
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h)
        return out

    def message(self, h_i, h_j):
        # Compute messages
        msg = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def aggregate(self, inputs, index):
        # Aggregate messages
        msg_aggr = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        return msg_aggr

    def update(self, aggr_out, h):
        upd_out = self.mlp_upd(torch.cat([h, aggr_out], dim=-1))
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class NewExpNormalSmearing(nn.Module):
    '''
    modified: delete cutoff with r
    '''
    def __init__(self, num_rbf, rbound_upper, rbound_lower=0.0, rbf_trainable=False, **kwargs):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = rbound_lower
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()
        if rbf_trainable:
            self.register_parameter("means", nn.parameter.Parameter(means))
            self.register_parameter("betas", nn.parameter.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        return torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class SincRadialBasis(nn.Module):
    # copied from Painn
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False, **kwargs):
        super().__init__()
        if rbf_trainable:
            self.register_parameter("n", nn.parameter.Parameter(torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0)/rbound_upper))
        else:
            self.register_buffer("n", torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0)/rbound_upper)

    def forward(self, r):
        n = self.n
        output = (math.pi) * n * torch.sinc(n * r)
        return output