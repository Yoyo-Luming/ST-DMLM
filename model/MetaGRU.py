import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
    

class GRU_Cell(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_meta):
        super(GRU_Cell, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_meta = dim_meta

    def forward(self, x, state, Wx, Wh, b):
        '''
        :param x: graph feature/signal - [B, N, C] or [B, B, H, H]
        :param state: previous hidden state - [B, N, H]
        :param Wx: weights - [B, N, C, 3*H]
        :param Wh: weights - [B, N, H, 3*H]
        :param b: weights - [B, N, 1, 3*H]
        :return h: current hidden state - [B, N, H]
        '''
        # print('x shape:', x.shape, state.shape, Wx.shape, Wh.shape, b.shape)
        if len(x.shape) == 3:
           x = torch.unsqueeze(x, -2) # B N 1 C
        # print('x shape:', x.shape)

        Wrx, Wzx, Wcx = torch.split(Wx, self.dim_hidden, dim=-1)
        Wrh, Wzh, Wch = torch.split(Wh, self.dim_hidden, dim=-1)
        br, bz, bc = torch.split(b, self.dim_hidden, dim=-1)     

        h = state.to(x.device)
        r = torch.sigmoid(x @ Wrx + h @ Wrh + br) # B N 1 H 
        z = torch.sigmoid(x @ Wzx + h @ Wzh + bz) # B N 1 H 
        hc = torch.tanh(x @ Wcx + (r*h) @ Wch + bc) # B N 1 H 

        h = (1-z) * h + z * hc # B N 1 H 
        return h
    
    def init_hidden(self, batch_size: int, num_nodes):
        return torch.zeros(batch_size, num_nodes, 1, self.dim_hidden)
    

class Encoder(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hidden, dim_meta, num_layers=1):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_in = dim_in
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.dim_meta = dim_meta
        # self.dim 
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_in if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(GRU_Cell(dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            dim_meta=self.dim_meta))
            
    def forward(self, x_seq, init_h, Wx, Wh, b):
        '''
        :param x_seq: graph feature/signal - [B, T, N, C]
        :param init_h: init hidden state - [B, N, H]*num_layers
        :param x_meta: meta - [B, N, H_M]
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
                h = self.cell_list[i](current_inputs[:, t, :, :], h, Wx, Wh, b) # B N H
                h_lst.append(h)
            output_h.append(h)
            current_inputs = torch.stack(h_lst, dim=1) # [B, T, N, H]
        return output_h
    
    def _init_hidden(self, batch_size: int):
        h_l = []
        for i in range(self.num_layers):
            h = self.cell_list[i].init_hidden(batch_size, self.num_nodes)
            h_l.append(h)
        return h_l
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class Decoder(nn.Module):
    def __init__(self, num_nodes, dim_out, dim_hidden, dim_meta, num_layers=1):
        super(Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.dim_out = dim_out
        self.dim_hidden = self._extend_for_multilayer(dim_hidden, num_layers)
        self.dim_meta = dim_meta
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.dim_out if i == 0 else self.dim_hidden[i - 1]
            self.cell_list.append(GRU_Cell(dim_in=cur_input_dim,
                                            dim_hidden=self.dim_hidden[i],
                                            dim_meta=self.dim_meta))
    
    def forward(self, x_t, h, Wx, Wh, b):
        '''
        :param x_t: graph feature/signal - [B, N, C]
        :param h: previous hidden state from the last encoder cell - [B, N, H]*num_layers
        :param x_meta: meta - [B, N, H_M]
        :return output: the last hidden state - [B, N, C]
        :return h_lst: hidden state of each layer - [B, N, H]*num_layers
        '''
        current_inputs = x_t # B N C
        h_lst= []
        for i in range(self.num_layers):
            h_t = self.cell_list[i](current_inputs, h[i], Wx, Wh, b)
            h_lst.append(h_t)
            current_inputs = h_t
        output = current_inputs 
        return output, h_lst
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class MetaGRU(nn.Module):
    def __init__(self, device, num_nodes, input_dim, output_dim, learner_hidden_dim, meta_dim, horizon, rnn_units, num_layers=1, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MetaGRU, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_dim = meta_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.decoder_dim = self.rnn_units
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        self.learner_wx = nn.Sequential(
            nn.Linear(self.meta_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * input_dim * rnn_units),
        )

        self.learner_wh = nn.Sequential(
            nn.Linear(self.meta_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * rnn_units * rnn_units),
        )

        self.learner_b = nn.Sequential(
            nn.Linear(self.meta_dim, learner_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(learner_hidden_dim, 3 * rnn_units),
        )

        self.encoder = Encoder(num_nodes=self.num_nodes, dim_in=self.input_dim, dim_hidden=self.rnn_units, dim_meta= self.meta_dim, num_layers=self.num_layers)
        self.decoder = Decoder(num_nodes=self.num_nodes, dim_out=self.input_dim, dim_hidden=self.decoder_dim, dim_meta= self.meta_dim, num_layers=self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, x, x_meta, labels=None, batches_seen=None):
        # x - [B, T, N, C]  x_meta - [N, H_M]
        batch_size = x.shape[0]
        x_meta = x_meta.expand(batch_size,self.num_nodes, self.meta_dim) # B N H_M
        Wx = self.learner_wx(x_meta).view(
            batch_size, self.num_nodes, self.input_dim, 3 * self.rnn_units
        )
        Wh = self.learner_wh(x_meta).view(
            batch_size, self.num_nodes, self.rnn_units, 3 * self.rnn_units
        )
        b = self.learner_b(x_meta).view(
            batch_size, self.num_nodes, 1, 3 * self.rnn_units
        )
        init_h = None

        h_lst = self.encoder(x, init_h, Wx, Wh, b) 

        go = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                                device=x.device)  # original initialization [B N C]
        out = list()
        for t in range(self.horizon):
            h_de, h_lst = self.decoder(go, h_lst, Wx, Wh, b)
            go = self.proj(torch.squeeze(h_de)) # 当Batchsize为 1 时 会将那当Batchsize那一维去掉，需要手动加回
            if batch_size == 1:
                go = torch.unsqueeze(go, 0) # B N 1 C
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
            
        # out: T*(B N C)
        outputs = torch.stack(out, dim=1) # B,T,N,C
        return outputs
    
# model = MetaGRU(device=torch.device('cpu'), num_nodes=207, input_dim=1, output_dim=1, meta_dim=32, horizon=12, rnn_units=64)
# from torchsummary import summary
# summary(model,[(64,12,207,1),(207,32)])