import torch
import torch.nn as nn
from models.encoder import Encoder
from torch.autograd import Function
from models.ae import AE

    

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.ae = AE(params)
        self.classifier = nn.Linear(256, self.params.num_of_classes) 


    def forward(self, x):
        if self.params.return_attention:
            recon, global_token,mu,attn_weights = self.ae(x) 
            return self.classifier(global_token), recon, mu,attn_weights 

        else:
            recon,global_token, mu = self.ae(x)
            return self.classifier(global_token), recon, mu

    def inference(self, x):
        if self.params.return_attention:
           
            mu, fused_glob, attn_e= self.ae.encoder(x) 
            return self.classifier(fused_glob),attn_e 

        else:

            mu,global_token = self.ae.encoder(x)
            return self.classifier(global_token)

            



