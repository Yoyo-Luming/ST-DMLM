import torch
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.empty(cheb_k * dim_in, dim_out), requires_grad=True)
        self.b = nn.Parameter(torch.empty(dim_out), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

    def forward(self, G, x):
        '''
        :param x: graph feature/signal - [B, N, C + H_in]
        :param G: support adj matrices - [K, N, N]
        :return output: hidden representation - [B, N, H_out]
        '''
        # print('x shape',x.shape, 'G shape', G.shape) # x shape torch.Size([64, 207, 65]) G shape torch.Size([6, 207, 207])
        support_list = list()
        for k in range(self.cheb_k):
            support = torch.einsum('ij,bjp->bip', [G[k, :, :], x]) # [B, N, C + H_in]
            support_list.append(support) # k*[B, N, C + H_in]
        support_cat = torch.cat(support_list, dim=-1) # [B, N, (C + H_in)*k]
        output = torch.einsum('bip,pq->biq', [support_cat, self.W]) +self.b # [B, N, H_out]
        # print('gcn output shape', output.shape) # gcn output shape torch.Size([64, 207, 256])
        return output
    

class LSTM_Cell(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, cheb_k):
        super(LSTM_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden

        self.conv_gate = GCN(cheb_k=cheb_k, dim_in=dim_in+dim_hidden, dim_out=4*dim_hidden)
    
    def forward(self, G, x_t, h_pre, c_pre):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x_t: graph feature/signal - [B, N, C]
        :param h_pre: previous hidden state - [B, N, H]
        :param c_pre: previous memory cell internal state - [B, N, H]
        :return h_t: current hidden state - [B, N, H]
        :return c_t: current memory cell internal state - [B, N, H]
        '''
        # print('x_t shape',x_t.shape, 'h_pre shape', h_pre.shape)
        combined = torch.cat([x_t, h_pre], dim=-1)
        combined_conv = self.conv_gate(G, combined)
        # print('combined_conv shape', combined_conv.shape) # combined_conv shape torch.Size([64, 207, 256])
        gc_i, gc_f, gc_o, gc_g = torch.split(combined_conv, self.dim_hidden, dim=-1)
        i = torch.sigmoid(gc_i) 
        f = torch.sigmoid(gc_f)
        o = torch.sigmoid(gc_o)
        g = torch.tanh(gc_g)
        # print('gate shape', f.shape, 'c shape', c_t_1.shape) # gate shape torch.Size([64, 207, 64]) c shape torch.Size([64, 207, 64])
        c_t = f * c_pre + i * g
        h_t = o * torch.tanh(c_t)
        # print('cell output h_t shape', h_t.shape)
        return h_t, c_t
    
    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        h = (weight.new_zeros(batch_size, self.num_nodes, self.dim_hidden))
        c = (weight.new_zeros(batch_size, self.num_nodes, self.dim_hidden))
        return h ,c
    

class Encoder(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, cheb_k, num_layers=1):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_in if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(LSTM_Cell(num_nodes=num_nodes,
                                            dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            cheb_k=cheb_k))
            
    def forward(self, G, x_seq, init_h, init_c):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x_seq: graph feature/signal - [B, T, N, C]
        :param init_h: init hidden state - [B, N, H]*num_layers
        :param init_c: init memory cell internal state - [B, N, H]*num_layers
        :return output_h: the last hidden state - [B, N, H]*num_layers
        :return output_c: the last memory cell internal state - [B, N, H]*num_layers
        '''
        batch_size, seq_len = x_seq.shape[:2]
        if init_h is None:
            init_h, init_c = self._init_hidden(batch_size)

        # print('encoder input shape', x_seq.shape) #encoder input shape torch.Size([64, 12, 207, 1])
        current_inputs = x_seq
        output_h,output_c = [], []
        for i in range(self.num_layers):
            h,c = init_h[i], init_c[i] 
            h_lst, c_lst = [], []
            for t in range(seq_len):
                h, c = self.cell_list[i](G, current_inputs[:, t, :, :], h, c) 
                h_lst.append(h)
                c_lst.append(c)
            output_h.append(h)
            output_c.append(c)
            current_inputs = torch.stack(h_lst, dim=1)
        return output_h, output_c
    
    def _init_hidden(self, batch_size: int):
        h_l = []
        c_l = []
        for i in range(self.num_layers):
            h , c = self.cell_list[i].init_hidden(batch_size)
            h_l.append(h)
            c_l.append(c)
        return h_l,c_l
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class Decoder(nn.Module):
    def __init__(self, num_nodes, dim_out, dim_hidden, cheb_k, num_layers=1):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_out = dim_out
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_out if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(LSTM_Cell(num_nodes=num_nodes,
                                            dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            cheb_k=cheb_k))
    
    def forward(self, G, x_t, h, c):
        '''
        :param G: support adj matrices - [K, N, N]
        :param x_t: graph feature/signal - [B, N, C]
        :param h: previous hidden state from the last encoder cell - [B, N, H]*num_layers
        :param c: previous memory cell internal state from the last encoder cell - [B, N, H]*num_layers
        :return output: the last hidden state - [B, N, C]
        :return h_lst: hidden state of each layer - [B, N, H]*num_layers
        :return c_lst: memory cell internal state of each layer - [B, N, H]*num_layers
        '''
        # print('decoder input shape', xt.shape) # decoder input shape torch.Size([64, 207, 1])
        current_inputs = x_t
        h_lst, c_lst = [],[]
        for i in range(self.num_layers):
            # print(i, 'layer', 'h_i shape ', h[i].shape, 'c_i shape ', c[i].shape)
            h_t, c_t = self.cell_list[i](G, current_inputs, h[i], c[i])
            h_lst.append(h_t)
            c_lst.append(c_t)
            current_inputs = h_t
        output = current_inputs
        return output, h_lst, c_lst
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class GCN_LSTM(nn.Module):
    def __init__(self, device, num_nodes, adj, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3):
        super(GCN_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.decoder_dim = self.rnn_units
        self.P = self.compute_cheby_poly(adj).to(device)  

        self.encoder = Encoder(num_nodes=self.num_nodes, dim_in=self.input_dim, dim_hidden=self.rnn_units, cheb_k=self.cheb_k, num_layers=self.num_layers)
        self.decoder = Decoder(num_nodes=self.num_nodes, dim_out=self.input_dim, dim_hidden=self.decoder_dim, cheb_k=self.cheb_k, num_layers=self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))

    def compute_cheby_poly(self, P: list):
        P_k = []
        for p in P:
            p = torch.from_numpy(p).float().T
            T_k = [torch.eye(p.shape[0]), p]    # order 0, 1
            for k in range(2, self.cheb_k):
                T_k.append(2*torch.mm(p, T_k[-1]) - T_k[-2])    # recurrent to order K
            P_k += T_k
        return torch.stack(P_k, dim=0)    # (K, N, N) or (2*K, N, N) for bidirection
    
    def forward(self, x):
        init_c = None
        init_h = None

        h_lst, c_lst = self.encoder(self.P, x,
                                  init_h, init_c) 

        deco_input = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                                 device=x.device)  # original initialization [B N C]
        outputs = list()
        for t in range(self.horizon):
            output, h_lst, c_lst = self.decoder(self.P, deco_input, h_lst, c_lst)
            deco_input = self.proj(output)  # update decoder input
            outputs.append(deco_input) # [B N C]*12

        outputs = torch.stack(outputs, dim=1) # B,T,N,C
        return outputs