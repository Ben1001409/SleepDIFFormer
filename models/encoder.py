from .layers.SelfAttention_Family import FullAttention, AttentionLayer,AttentionLayer_diff,FullAttention_diff
from .layers.Embed import DataEmbedding_inverted, PositionalEmbedding,DataEmbedding,PatchEmbedding,DataEmbedding_wo_pos
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, 30, d_model).normal_(std=0.02)
        ) 

        self.dropout = nn.Dropout(dropout)


        self.embed_decreasing_kernel_128 = nn.Sequential(
            nn.Conv1d(n_vars, 64, kernel_size=49, stride=1, padding=24),
            nn.BatchNorm1d(64), 
            nn.GELU(),
            nn.MaxPool1d(5,5),            

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128), 
            nn.GELU(),
            nn.MaxPool1d(5,5),             

            nn.Conv1d(128, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), 
            nn.GELU(),
            nn.MaxPool1d(4,4),            
        )

        
    def apply_rotary_pos_emb(x, seq_dim=-2):

        dim = x.shape[-1]
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, device=x.device).float()
        inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

        t = torch.arange(x.shape[seq_dim], device=x.device).float()
        sinusoid_inp = torch.einsum('i , j -> i j', t, inv_freq)  
        sin = sinusoid_inp.sin()[None, :, :]
        cos = sinusoid_inp.cos()[None, :, :]

        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated


    def forward(self, x):
        n_vars = x.shape[1]

        glb = self.glb_token.repeat((x.shape[0], 1, 1))
        x=self.embed_decreasing_kernel_128(x) 
        x=x.transpose(1,2)
        #print(x.shape)
        #print(glb.shape)

        x = x + self.pos_embedding
        x = torch.cat([x, glb], dim=1)
        return self.dropout(x), n_vars

    



class Transformer_Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Transformer_Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='gelu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)


        #self.apply(init_weights)
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        all_attn = {
        "eeg_self_attn": [],
        "eog_self_attn": [],
        "cross_attn_eeg_to_eog": [],
        "cross_attn_eog_to_eeg": []
    }
        for layer in self.layers:
            x_eeg,x_eog, attn_dict = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            for k in all_attn:
                all_attn[k].append(attn_dict[k])
            

        if self.norm is not None:
            x_eeg = self.norm(x)
            x_eog = self.norm(x)

        if self.projection is not None:
            x_eeg = self.projection(x)
            x_eog = self.projection(x)
        #return x_eeg,x_eog,last_self_attn
        #=all_attn['eeg_self_attn']
        #print(eeg_self_attn[0].shape)
        return x_eeg,x_eog,all_attn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model


        self.self_attention  = self_attention    
        self.cross_attention = cross_attention  
                                                 
                                                  
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,   out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff,   out_channels=d_model, kernel_size=1)
        self.mlp_feedforward = MLPBlock(d_model, d_model, dropout)

        
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  
        self.norm3 = nn.LayerNorm(d_model)  
        self.norm4 = nn.LayerNorm(d_model)  
        self.norm5 = nn.LayerNorm(d_model)  
        self.norm6 = nn.LayerNorm(d_model)
        self.norm7 = nn.LayerNorm(d_model)

        # self.norm1 = RMSNorm(d_model)
        # self.norm2 = RMSNorm(d_model)
        # self.norm3 = RMSNorm(d_model)
        # self.norm4 = RMSNorm(d_model)
        # self.norm5 = RMSNorm(d_model)
        # self.norm6 = RMSNorm(d_model)
        # self.norm7 = RMSNorm(d_model)


        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross_seq, x_mask=None, cross_mask=None, tau=None, delta=None):


        B, L_x, D = x.shape
        B, L_c, _ = cross_seq.shape


        cross_normed = self.norm5(cross_seq)  
        #cross_normed=cross_seq # noprenorm
        cross_input = cross_normed  
        cross_self_out, eog_self_attn = self.self_attention(
            cross_input,   
            cross_input,    
            cross_input,   
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )
        #print("eog self attn shape: ",eog_self_attn.shape) 
        cross_self_out = cross_self_out  

    
        cross_seq = cross_seq + self.dropout(cross_self_out)

        cross_glb_ori = cross_seq[:, -1:, :]   


        x_normed = self.norm4(x)          
        #x_normed=x # no prenorm
        x_input = x_normed 
        eeg_self_out, eeg_self_attn = self.self_attention(
            x_input, x_input, x_input,
            attn_mask=x_mask,
            tau=tau, delta=None
        )
        #print("eeg self attn shape: ",eeg_self_attn.shape) 
        eeg_self_out = eeg_self_out  
        x = x + self.dropout(eeg_self_out)          
        x = self.norm1(x)                            


        eeg_glb_ori = x[:, -1:, :]         
 
        eeg_glb_input = eeg_glb_ori 
        eog_glb_input = cross_glb_ori 

        #print(eeg_eog_cross_attn.shape)
        cross_out_eeg, eeg_cross_attn = self.cross_attention(
            eeg_glb_input,    
            eog_glb_input,      
            eog_glb_input,    
            attn_mask=None,
            tau=tau, delta=delta
        )
        #print("cross attn eeg shape: ",eeg_cross_attn.shape)
        cross_out_eog, eog_cross_attn = self.cross_attention(
            eog_glb_input,    
            eeg_glb_input,     
            eeg_glb_input,    
            attn_mask=None,
            tau=tau, delta=delta
        )
        #print("cross attn eog shape: ",eog_cross_attn.shape) 

        eog_glb=cross_glb_ori +self.dropout(cross_out_eog)
        eog_glb=self.norm6(eog_glb)
        #cross_out = cross_out 

        eeg_glb = eeg_glb_ori + self.dropout(cross_out_eeg)  
        eeg_glb = self.norm2(eeg_glb)

        x_patches_eeg = x[:, :-1, :]           


        #concatenate
        x_eeg = torch.cat([ x_patches_eeg,    
                        eeg_glb,        
                      ], dim=1)

        x_patches_eog=cross_seq[:, :-1, :]
        x_eog = torch.cat([ x_patches_eog,    
                        eog_glb,       
                      ], dim=1)


        y_eeg = x_eeg.transpose(1, 2)  
        y_eeg = self.dropout(self.activation(self.conv1(y_eeg)))  
        y_eeg = self.dropout(self.conv2(y_eeg).transpose(1, 2))   

        y_eog = x_eog.transpose(1, 2) 
        y_eog = self.dropout(self.activation(self.conv3(y_eog)))  
        y_eog = self.dropout(self.conv4(y_eog).transpose(1, 2))

       
        
        out_eeg = self.norm3(x_eeg + y_eeg)  
        out_eog = self.norm7(x_eog + y_eog)

        #return out_eeg,out_eog, attn_weights
    #     return out_eeg,out_eog,{
    #     "eeg_self_attn": eeg_self_attn,
    #     "eog_self_attn": eog_self_attn,
    #     "cross_attn_eeg_to_eog": eeg_cross_attn,
    #     "cross_attn_eog_to_eeg": eog_cross_attn
    # }
        return out_eeg,out_eog,{
            "eeg_self_attn": eeg_self_attn,
            "eog_self_attn": eog_self_attn,
            "cross_attn_eeg_to_eog": eeg_cross_attn,
            "cross_attn_eog_to_eeg": eog_cross_attn,
            
        }


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.use_norm=True
        self.n_vars=1

        self.en_embedding_1 = EnEmbedding(self.n_vars, d_model=self.params.d_model, patch_len=self.params.patch_len, dropout=0.1)
        self.en_embedding_2 = EnEmbedding(self.n_vars, d_model=self.params.d_model, patch_len=self.params.patch_len, dropout=0.1)
        
        if not self.params.use_normal:
            print("use diff: ",not self.params.use_normal)
            self.seq_encoder_1=Transformer_Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer_diff(
                                FullAttention_diff(l,d_model=self.params.d_model,n_heads=self.params.num_heads, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True,mask_flag=False),
                                l, d_model=self.params.d_model, n_heads=self.params.num_heads),
                            AttentionLayer_diff(
                                FullAttention_diff(l,d_model=self.params.d_model,n_heads=self.params.num_heads, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True,mask_flag=False),
                                l,d_model=self.params.d_model, n_heads=self.params.num_heads),
                            d_model=self.params.d_model,
                            d_ff=self.params.d_ff,
                            dropout=self.params.dropout,
                            activation='gelu',
                        )
                        for l in range(self.params.num_layers)
                    ],
            )
            self.seq_encoder_2=Transformer_Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer_diff(
                                FullAttention_diff(l,d_model=self.params.d_model,n_heads=self.params.num_heads, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True,mask_flag=False),
                                l, d_model=self.params.d_model, n_heads=self.params.num_heads),
                            AttentionLayer_diff(
                                FullAttention_diff(l,d_model=self.params.d_model,n_heads=self.params.num_heads, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True,mask_flag=False),
                                l,d_model=self.params.d_model, n_heads=self.params.num_heads),
                            d_model=self.params.d_model,
                            d_ff=self.params.d_ff,
                            dropout=self.params.dropout,
                            activation='gelu',
                        )
                        for l in range(self.params.num_layers)
                    ],
            )
        else:
            print("use normal: ",self.params.use_normal)
            self.seq_encoder_1=Transformer_Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True),
                                d_model=self.params.d_model, n_heads=self.params.num_heads),
                            AttentionLayer(
                                FullAttention(False, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True),
                                d_model=self.params.d_model, n_heads=self.params.num_heads),
                            d_model=self.params.d_model,
                            d_ff=self.params.d_ff,
                            dropout=self.params.dropout,
                            activation='gelu',
                        )
                        for l in range(self.params.num_layers)
                    ],
            )

            self.seq_encoder_2=Transformer_Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=True),
                                d_model=self.params.d_model, n_heads=self.params.num_heads),
                            AttentionLayer(
                                FullAttention(False, factor=1, attention_dropout=self.params.dropout,
                                            output_attention=False),
                                d_model=self.params.d_model, n_heads=self.params.num_heads),
                            d_model=self.params.d_model,
                            d_ff=self.params.d_ff,
                            dropout=self.params.dropout,
                            activation='gelu',
                        )
                        for l in range(self.params.num_layers)
                    ],
            )

        self.fc_mu1 = nn.Linear(256,256)
        self.seq_encoder_global1 = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=256,
            depth=0,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
            return_attention=self.params.return_attention,
        )
        self.seq_encoder_global2 = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=256,
            depth=0,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
            return_attention=self.params.return_attention,
        )
        self.mlp = MLPBlock(512, 512, 0.1)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)

    
    ##no flipping, cross attention between global token
    def forward(self, x):

        B, E, C, T = x.shape
        assert C == 2

        eeg = x[:, :, 0:1, :].view(B * E, 1, T)  
        eog = x[:, :, 1:2, :].view(B * E, 1, T) 


        eeg_tokens,_ = self.en_embedding_1(eeg)
        eog_tokens,_ = self.en_embedding_2(eog)


        
        enc_eeg,enc_eog, all_attns = self.seq_encoder_1(
            eeg_tokens,
            eog_tokens
        )


        glob_eeg = enc_eeg[:, -1, :]        
        out_eeg  = enc_eeg[:, :-1, :]   

        glob_eog = enc_eog[:, -1, :]        
        out_eog  = enc_eog[:, :-1, :]      
        
        patches = torch.cat([out_eeg, out_eog], dim=-1)
       
        fused_glob = torch.cat([glob_eeg, glob_eog], dim=-1)
        patches = patches.transpose(2, 1)       

        pooled  = F.adaptive_avg_pool1d(patches, 1).squeeze(-1)  
        pooled  = pooled.view(B, E, -1)       
    
        pooled_enc,_ = self.seq_encoder_global1(pooled) 
        mu = self.fc_mu1(pooled_enc)   

        fused_glob = fused_glob.view(B, E, -1)
        #print(fused_glob.shape)    
        glob_enc,_   = self.seq_encoder_global2(fused_glob)  
        fused_out  = self.fc_mu1(glob_enc)      
        return mu, fused_out, all_attns


