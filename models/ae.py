import torch
import torch.nn as nn
from models.transformer import TransformerEncoder
from models.encoder import Encoder
from models.decoder import Decoder



class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        self.params=params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.params.return_attention:
            mu,global_token,attn_weights = self.encoder(x) 
            
        else:
            mu,global_token = self.encoder(x) 
       
        recon = self.decoder(mu)
        if self.params.return_attention:
            return recon,global_token,mu,attn_weights   
        else:
            return recon,global_token,mu 


