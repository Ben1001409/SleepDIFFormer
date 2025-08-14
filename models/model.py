import torch
import torch.nn as nn
from models.encoder import Encoder
from torch.autograd import Function
from models.ae import AE

class SequenceMLPClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        hidden_sizes: list = [256, 128,64,32],
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_features = d_model

        for hs in hidden_sizes:
            layers.append(nn.Linear(in_features, hs))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_features = hs

        layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:

        logits = self.classifier(encoder_output)
        return logits
    

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.ae = AE(params)
        self.classifier = nn.Linear(256, self.params.num_of_classes)

        self.mlp_classifier=SequenceMLPClassifier(512,[256,128],self.params.num_of_classes,0.1)

    def forward(self, x):
        if self.params.return_attention:
            recon, global_token,mu,attn_weights = self.ae(x) 

            return self.classifier(global_token), recon, mu,attn_weights 

        else:
            recon, mu = self.ae(x)
            return self.classifier(mu), recon, mu
            #return self.mlp_classifier(mu), recon, mu

    def inference(self, x):
        if self.params.return_attention:

            mu, fused_glob, attn_e= self.ae.encoder(x) 
            return self.classifier(fused_glob),attn_e 

        else:

            mu,global_token = self.ae.encoder(x)

            return self.classifier(global_token)

