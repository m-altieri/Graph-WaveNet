import torch
import torch.nn as nn
from rgsl.model.RGCN import AVWGCN


class RGSLCell(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(RGSLCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(cheb_polynomials, L_tilde, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(cheb_polynomials, L_tilde, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, learned_tilde):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        #print(f'input_and_state: {input_and_state}')
        #print(f'node_embeddings: {node_embeddings}')
        #print(f'learned_tilde: {learned_tilde}')

        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde))
        #print(f'z_r: {z_r}')
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        #print(f'candidate: {candidate}')
        hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde))
        #print(f'hc: {hc}')
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
