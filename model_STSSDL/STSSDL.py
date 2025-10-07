import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, num_support):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(num_support*cheb_k*dim_in, dim_out)) # num_support*cheb_k*dim_in is the length of support
        # self.weights = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        for support in supports:
            if len(support.shape) == 2:
                support_ks = [torch.eye(support.shape[0]).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                support_ks = [torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_support):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k, num_support)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Encoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, x, init_state, supports):
        #shape of x: (B, T, N, D), shape of init_state: (rnn_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (rnn_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Decoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (rnn_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.rnn_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class STSSDL(nn.Module):
    def __init__(self, num_nodes=207, input_dim=1, output_dim=1, horizon=12, rnn_units=128, rnn_layers=1, cheb_k=3,
                 ycov_dim=1, prototype_num=20, prototype_dim=64, tod_embed_dim=10, adj_mx=None, cl_decay_steps=2000,
                 TDAY=288, use_curriculum_learning=True, use_STE=False, device="cpu",adaptive_embedding_dim=48,node_embedding_dim=20,input_embedding_dim=128):
        super(STSSDL, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.tod_embed_dim = tod_embed_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.device = device
        self.use_STE = use_STE
        self.TDAY = TDAY
        self.adaptive_embedding_dim=adaptive_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.input_embedding_dim=input_embedding_dim
        self.total_embedding_dim=  self.tod_embed_dim+self.adaptive_embedding_dim+self.node_embedding_dim
        # prototypes
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.prototypes = self.construct_prototypes()
        
        # projection & spatio-temporal embedding
        if self.use_STE:
            if self.adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(12, num_nodes, self.adaptive_embedding_dim))
                )
            self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)
            self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(self.TDAY, self.tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        # encoder
        self.adj_mx = adj_mx
        if self.use_STE:
            self.encoder = ADCRNN_Encoder(self.num_nodes,  input_embedding_dim + self.total_embedding_dim, self.rnn_units, self.cheb_k, self.rnn_layers, len(self.adj_mx))
        else:
            self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers, len(self.adj_mx))
        
        # decoder
        self.decoder_dim = self.rnn_units + self.prototype_dim
        if self.use_STE:
            self.decoder = ADCRNN_Decoder(self.num_nodes, input_embedding_dim + self.total_embedding_dim-self.adaptive_embedding_dim, self.decoder_dim, self.cheb_k, self.rnn_layers, 1)
        else:
            self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.rnn_layers, 1)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
        # graph
        self.hypernet = nn.Sequential(nn.Linear(self.decoder_dim*2, self.tod_embed_dim, bias=True))
        
        self.act_dict = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(), 'sigmoid': nn.Sigmoid()}
        self.act_fn = 'sigmoid'  # 'relu' 'lrelu' 'sigmoid'
        
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_prototypes(self):
        prototypes_dict = nn.ParameterDict()
        prototype = torch.randn(self.prototype_num, self.prototype_dim)
        prototypes_dict['prototypes'] = nn.Parameter(prototype, requires_grad=True)     # (M, d)
        prototypes_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.prototype_dim), requires_grad=True)    # project to query
        for param in prototypes_dict.values():
            nn.init.xavier_normal_(param)

        return prototypes_dict
    
    def query_prototypes(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.prototypes['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.prototypes['prototypes'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.prototypes['prototypes'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes['prototypes'][ind[:, :, 0]] # B, N, d
        neg = self.prototypes['prototypes'][ind[:, :, 1]] # B, N, d
        mask = torch.stack([ind[:, :, 0], ind[:, :, 1]], dim=-1) # B, N, 2
            
        return value, query, pos, neg, mask
    
    def calculate_distance(self, pos, pos_his, mask=None):
        score = torch.sum(torch.abs(pos - pos_his), dim=-1)
        return score, mask
            
    def forward(self, x, x_cov, x_his, y_cov, labels=None, batches_seen=None):
        if self.use_STE:
            if self.input_embedding_dim>0:
                x = self.input_proj(x)  # [B,T,N,1]->[B,T,N,D]
            features = [x]

            tod = x_cov.squeeze()  # [B, T, N]
            if self.tod_embed_dim>0:
                time_emb = self.time_embedding[(x_cov.squeeze() * self.TDAY).type(torch.LongTensor)]  # [B, T, N, d]
                features.append(time_emb)
            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(
                    size=(x.shape[0], *self.adaptive_embedding.shape)
                )
                features.append(adp_emb)
            if self.node_embedding_dim>0:
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(x.shape[0], self.horizon, -1, -1)  # [B,T,N,d]
                features.append(node_emb)
            x = torch.cat(features, dim=-1) # [B, T, N, D+d+80]
        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)    
        v_t, q_t, p_t, n_t, mask = self.query_prototypes(h_t)
        if self.use_STE:
            if self.input_embedding_dim>0:
                x_his = self.input_proj(x_his)  # [B,T,N,1]->[B,T,N,D]
            features = [x_his]
            tod = x_cov.squeeze()  # [B, T, N]
            if self.tod_embed_dim>0:
                time_emb = self.time_embedding[(x_cov.squeeze() * self.TDAY).type(torch.LongTensor)]  # [B, T, N, d]

                features.append(time_emb)
            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(
                        size=(x.shape[0], *self.adaptive_embedding.shape)
                    )
                features.append(adp_emb)
            if self.node_embedding_dim>0:
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(x.shape[0], self.horizon, -1, -1)  # [B,T,N,d]
                features.append(node_emb)
            x_his = torch.cat(features, dim=-1) # [B, T, N, D+d+80]
        h_his_en, state_his_en = self.encoder(x_his, init_state, supports_en) # B, T, N, hidden
        h_a = h_his_en[:, -1, :, :] # B, N, hidden (last state)      
        v_a, q_a, p_a, n_a, mask_his = self.query_prototypes(h_a)
        
        latent_dis, _ = self.calculate_distance(q_t, q_a)
        prototype_dis, mask_dis = self.calculate_distance(p_t, p_a)
        
        query = torch.stack([q_t, q_a], dim=0)
        pos = torch.stack([p_t, p_a], dim=0)
        neg = torch.stack([n_t, n_a], dim=0)
        mask = torch.stack([mask, mask_his], dim=0) if mask is not None else [None, None] 
        
        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1) # B, N, D
        
        node_embeddings = self.hypernet(h_aug) # B, N, e
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1) 
        supports_de = [support]
        
        ht_list = [h_de]*self.rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        
        out = []
        for t in range(self.horizon):
            if self.use_STE:
                if self.input_embedding_dim>0:
                    go = self.input_proj(go)  # equal to torch.zeros(B,N,D)
                features = [go]
                tod = y_cov[:, t, ...].squeeze()  # [B, T, N]
                if self.tod_embed_dim>0:
                    time_emb = self.time_embedding[(tod * self.TDAY).type(torch.LongTensor)]
                    features.append(time_emb)
                if self.node_embedding_dim>0:
                    node_emb = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B,N,d]
                    features.append(node_emb)
                go = torch.cat(features, dim=-1) # [B, T, N, D+d]
                h_de, ht_list = self.decoder(go, ht_list, supports_de)
            else:
                h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports_de)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
                    
        output = torch.stack(out, dim=1)
        
        return output, query, pos, neg, mask, latent_dis, prototype_dis

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
    return

def main():
    from torchinfo import summary
    from utils import load_adj
    
    adj_mx = load_adj('../METRLA/adj_mx.pkl', "symadj")
    adj_mx = [torch.FloatTensor(i) for i in adj_mx]
    model = STSSDL(adj_mx=adj_mx)
    summary(model, [[8, 12, 207, 1], [8, 12, 207, 1], [8, 12, 207, 1], [8, 12, 207, 1]], device="cpu")
    
if __name__ == '__main__':
    main()
