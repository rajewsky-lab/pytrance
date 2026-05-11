import torch.nn as nn
from ..layers import GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h, act='sigmoid', layer_type='gcn'):
        #super(DGI, self).__init__()
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'gcn':
            layers = [GCN(n_in, n_h[0], act)]
            for l in range(1, len(n_h)):
                layers.append(GCN(n_h[l-1], n_h[l], act))
        else:
            raise ValueError(f'{layer_type} layer not implemented')
        
        self.layers = nn.Sequential(*layers)
        
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h[-1])

    def forward(self, seq1, adj1, seq2, adj2, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.layers[0](seq1, adj1, sparse)
        h_2 = self.layers[0](seq2, adj2, sparse)
        for l in range(1, len(self.layers)):
            h_1 = self.layers[l](h_1, adj1, sparse)
            h_2 = self.layers[l](h_2, adj2, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.layers[0](seq, adj, sparse)
        for l in range(1, len(self.layers)):
            h_1 = self.layers[l](h_1, adj, sparse)
            
        c = self.read(h_1, msk)
        c = self.sigm(c)

        return h_1.detach(), c.detach()

