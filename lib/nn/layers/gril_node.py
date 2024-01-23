import torch
import torch.nn as nn
from einops import rearrange
from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor
from .gtn import GTN
from .nodeformer import NodeFormerConv
from .nagphormer import EncoderLayer, laplacian_positional_encoding
import dgl
from dgl.data import DGLDataset
import scipy.sparse as sp

class DGLGraphDataset(DGLDataset):
    def __init__(self, adjacency_matrices):
        super(DGLGraphDataset, self).__init__(name='dgl_graph_dataset', raw_dir=None, save_dir=None, force_reload=False)
        self.adjacency_matrices = adjacency_matrices

    def process(self):
        self.dgl_graphs = []
        for adj_matrix in self.adjacency_matrices:
            graph = self.adjacency_matrix_to_dgl_graph(adj_matrix)
            self.dgl_graphs.append(graph)

    def __getitem__(self, idx):
        return self.dgl_graphs[idx]

    def __len__(self):
        return len(self.adjacency_matrices)

    def adjacency_matrix_to_dgl_graph(self, adj_matrix):
        if torch.sparse.is_sparse_tensor(adj_matrix):
            # Convert sparse tensor to dense numpy array
            adj_matrix_np = adj_matrix.to_dense().numpy()
            # Convert the numpy array to a DGL graph
            graph = dgl.from_scipy(sp.csr_matrix(adj_matrix_np))
        else:
            # Convert dense tensor to a DGL graph
            graph = dgl.from_scipy(sp.csr_matrix(adj_matrix.numpy()))
        return graph


class SpatialDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.):
        super(SpatialDecoder, self).__init__()
        ## d_in 66
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model,
                                            support_len=support_len * order, order=1, include_self=False)
        self.nodeconv = NodeFormerConv(32, 32,
                                       num_heads=2, 
                                       nb_random_features=4,
                                       use_gumbel=True, 
                                       nb_gumbel_sample=16, 
                                       rb_order=1, 
                                       rb_trans='sigmoid')
        self.encode = EncoderLayer(32,32,
                                   dropout_rate = 0.1, 
                                   attention_dropout_rate = 0.1, 
                                   num_heads = 8)
        
        self.register_parameter('spatial_att', None)
        self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None

    def forward(self, x, m, h, u, adj, cached_support=False):
        # [batch, channels, nodes]
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        x_in = self.lin_in(x_in)

        out = self.graph_conv(x_in, adj)
        ## SPATIAL ATTENTION
        out = torch.cat([out, h], 1)
        
        out = self.activation(self.lin_out(out))
        # out = self.lin_out(out)
        out = torch.cat([out, h], 1)
        return self.read_out(out), out

class GRILNODE(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 support_len=2,
                 n_nodes=None,
                 layer_norm=False):
        super(GRILNODE, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
    
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)
        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)
        self.nodeconv = NodeFormerConv(2, 2,
                                       num_heads=4, 
                                       nb_random_features=7,
                                       use_gumbel=True, 
                                       nb_gumbel_sample=30, 
                                       rb_order=1, 
                                       rb_trans='sigmoid')

        self.encode = EncoderLayer(32,32,
                                   dropout_rate = 0.1, 
                                   attention_dropout_rate = 0.1, 
                                   num_heads = 8)
        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        #####################################
        # adj_matrix_np = adj[0].numpy()
        # graph = dgl.from_scipy(sp.csr_matrix(adj_matrix_np))
        # graph = dgl.to_bidirected(graph)
        # lpe = laplacian_positional_encoding(graph, pos_enc_dim=3) 
        
        # x_prime = x.permute(0,2,1)
        # features = []
        # for i in range(x_prime.shape[0]):
        #     feature = torch.cat((x_prime[i], lpe), dim=1)
        #     features.append(feature)
        # features = torch.stack(features, dim=0).squeeze(1)

        # output = self.encode(features)
        # print(output.shape)
        # exit()

        #####################################

        # index = adj[0].nonzero().t().contiguous()
        # edge_index = [[]]
        # edge_index[0].append(index[0])
        # edge_index[0].append(index[1])
        # imputation = []

        # for i in range(rnn_in.shape[0]):
        #     prediction = self.nodeconv(rnn_in[i].permute(1,0).unsqueeze(0), edge_index, tau=1)
        #     imputation.append(prediction[0])
        # rnn_in = torch.stack(imputation, dim=0).squeeze(1).permute(0,2,1)
    
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        # x:[batch, features, nodes, steps]
        # print(dgl.DGLGraph(adj[0]))
        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]
        # Temporal conv
        predictions, imputations, states = [], [], []
        representations = []
       
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None

            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)  #[8,1,20]
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, 
                                                    adj=adj, cached_support=cached_support)  # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj)
         
            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)
    
        imputations = torch.stack(imputations, dim=-1) ## [16,1,20,12]
        predictions = torch.stack(predictions, dim=-1) ## [16,1,20,12]
   
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states
        # return imputations

class BiGRILNODE(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                ff_size,
                ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(BiGRILNODE, self).__init__()
        self.fwd_rnn = GRILNODE(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)
        self.bwd_rnn = GRILNODE(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)
        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        # Forward
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation_repr = imputation
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions,fwd_repr,bwd_repr,imputation_repr


