import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, K):
        super(GCN, self).__init__()
        self.K = K
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.FloatTensor(K*dim_in, dim_out))
        self.b = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

    def forward(self, G, x):
        '''
        :param x: graph feature/signal - [B, N, C + H_in]
        :param G: support adj matrices - [K, N, N]
        :return output: hidden representation - [B, N, H_out]
        '''
        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [G[k, :, :], x]) # [B, N, C + H_in]
            support_list.append(support) # k*[B, N, C + H_in]
        support_cat = torch.cat(support_list, dim=-1) # [B, N, (C + H_in)*k]
        output = torch.einsum('bip,pq->biq', [support_cat, self.W]) +self.b # [B, N, H_out]
        return output
    

class GRU_Cell(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, K):
        super(GRU_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden

        self.conv_gate = GCN(dim_in=dim_in+dim_hidden, dim_out=2*dim_hidden, K=K)
        self.update = GCN(dim_in=dim_in+dim_hidden, dim_out=dim_hidden, K=K)
    
    def forward(self, G, x, state):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x: graph feature/signal - [B, N, C]
        :param state: previous hidden state - [B, N, H]
        :return h: current hidden state - [B, N, H]
        '''
        state = state.to(x.device)
        combined = torch.cat([x, state], dim=-1)

        combined_conv = torch.sigmoid(self.conv_gate(G, combined))
        z, r = torch.split(combined_conv, self.dim_hidden, dim=-1) # B N H 
        candidate = torch.cat((x,r*state), dim=-1)
        hc = torch.tanh(self.update(G, candidate)) # B N H 
        h = (1-z)*state + z*hc # B N H 
        return h
    
    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, self.num_nodes, self.dim_hidden)
    

class Encoder(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, K, num_layers=1):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_in if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(GRU_Cell(num_nodes=num_nodes,
                                            dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            K=K))
            
    def forward(self, G, x_seq, init_h):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x_seq: graph feature/signal - [B, T, N, C]
        :param init_h: init hidden state - [B, N, H]*num_layers
        :return output_h: the last hidden state - [B, N, H]*num_layers
        '''
        batch_size, seq_len = x_seq.shape[:2]
        if init_h is None:
            init_h = self._init_hidden(batch_size)
        current_inputs = x_seq
        output_h = []
        for i in range(self.num_layers):
            h = init_h[i]
            h_lst= []
            for t in range(seq_len):
                h = self.cell_list[i](G, current_inputs[:, t, :, :], h) # B N C or H
                h_lst.append(h)
            output_h.append(h)
            current_inputs = torch.stack(h_lst, dim=1)
        return output_h
    
    def _init_hidden(self, batch_size: int):
        h_l = []
        for i in range(self.num_layers):
            h = self.cell_list[i].init_hidden(batch_size)
            h_l.append(h)
        return h_l
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class Decoder(nn.Module):
    def __init__(self, num_nodes, dim_out, dim_hidden, K, num_layers=1):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_out = dim_out
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_out if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(GRU_Cell(num_nodes=num_nodes,
                                            dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            K=K))
    
    def forward(self, G, x_t, h):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x_t: graph feature/signal - [B, N, C]
        :param h: previous hidden state from the last encoder cell - [B, N, H]*num_layers
        :return output: the last hidden state - [B, N, C]
        :return h_lst: hidden state of each layer - [B, N, H]*num_layers
        '''
        current_inputs = x_t
        h_lst= []
        for i in range(self.num_layers):
            h_t = self.cell_list[i](G, current_inputs, h[i])
            h_lst.append(h_t)
            current_inputs = h_t
        output = current_inputs
        return output, h_lst
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class GCN_GRU(nn.Module):
    def __init__(self, device, num_nodes, adj, input_dim, output_dim, horizon, rnn_units, num_layers=1, K=3, cl_decay_steps=2000, use_curriculum_learning=True):
        super(GCN_GRU, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.K = K
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.decoder_dim = self.rnn_units
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.P = self.compute_cheby_poly(adj).to(device)  

        self.encoder = Encoder(num_nodes=self.num_nodes, dim_in=self.input_dim, dim_hidden=self.rnn_units, K=self.K, num_layers=self.num_layers)
        self.decoder = Decoder(num_nodes=self.num_nodes, dim_out=self.input_dim, dim_hidden=self.decoder_dim, K=self.K, num_layers=self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))

    def compute_cheby_poly(self, P: list):
        P_k = []
        for p in P:
            p = torch.from_numpy(p).float().T
            T_k = [torch.eye(p.shape[0]), p]    # order 0, 1
            for k in range(2, self.K):
                T_k.append(2*torch.mm(p, T_k[-1]) - T_k[-2])    # recurrent to order K
            P_k += T_k
        return torch.stack(P_k, dim=0)    # (K, N, N) or (2*K, N, N) for bidirection
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, x, labels=None, batches_seen=None):
        init_h = None

        h_lst = self.encoder(self.P, x,
                                  init_h) 

        go = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                                 device=x.device)  # original initialization [B N C]
        out = list()

        for t in range(self.horizon):
            h_de, h_lst = self.decoder(self.P, go, h_lst)
            go = self.proj(h_de) # B N C 
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        outputs = torch.stack(out, dim=1) # B,T,N,C
        return outputs