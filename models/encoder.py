from .layers.SelfAttention_Family import FullAttention, AttentionLayer,AttentionLayer_diff,FullAttention_diff
from .layers.Embed import DataEmbedding_inverted, PositionalEmbedding,DataEmbedding,PatchEmbedding,DataEmbedding_wo_pos
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, patch_num, num_classes=5, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, d_model)) 
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_vars * d_model, num_classes)

    def forward(self, x):  
        x = self.pool(x.permute(0, 1, 3, 2))  
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)  



class ThreeLayerNonOverlappingPatchConv(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=5, padding=0)  
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=0)           
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=2, stride=2, padding=0)    

        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self, x): 
        x = self.act(self.bn1(self.conv1(x))) 
        x = self.act(self.bn2(self.conv2(x))) 
        x = self.act(self.bn3(self.conv3(x))) 
        return x             

class SpectralPatchMaxPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.spec_convs = nn.ModuleList([
            nn.Conv1d(1, 32, k, padding=k//2, bias=False) for k in (3, 9, 27)
        ])
        self.bn_spec = nn.BatchNorm1d(96)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2, 2)
        self.patch_conv = nn.Conv1d(96, d_model, 25, stride=25, bias=False)
        self.bn_patch = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.spec_convs], dim=1)
        x = self.act(self.bn_spec(x))
        x = self.pool(x)
        x = self.act(self.bn_patch(self.patch_conv(x)))
        return x


class MultiScalePatchEmbed(nn.Module):
    def __init__(self, d_model, patch_scales=[25,50,75]):
        super().__init__()
        self.spec_convs = nn.ModuleList([
            nn.Conv1d(1, d_model // len(patch_scales), kernel_size=1)  
            for _ in patch_scales
        ])
        self.patch_convs = nn.ModuleList()
        for scale in patch_scales:
            self.patch_convs.append(
                nn.Conv1d(d_model//len(patch_scales), d_model//len(patch_scales),
                          kernel_size=scale, stride=scale, bias=False)
            )

    def forward(self, x):
        all_tokens = []
        for proj, patch in zip(self.spec_convs, self.patch_convs):
            y = proj(x)                
            y = patch(y)               
            all_tokens.append(y)       
        x = torch.cat(all_tokens, dim=1)  
        
        return x  


class SpectralFilterBank(nn.Module):
    def __init__(self, in_ch=1, out_ch_per_filter=32, kernel_sizes=(3,9,27)):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch_per_filter, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_ch_per_filter * len(kernel_sizes))
        self.act = nn.GELU()

    def forward(self, x): 
        filtered = [f(x) for f in self.filters]       
        x = torch.cat(filtered, dim=1)                
        x = self.act(self.bn(x))
        return x

class SpectralPatch(nn.Module):
    def __init__(self, d_model, patch_size):
        super().__init__()
        self.spec_bank = SpectralFilterBank(1, out_ch_per_filter=32, kernel_sizes=(3,9,27))
        self.patch_conv = nn.Conv1d(32*3, d_model, patch_size, stride=patch_size, bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.spec_bank(x)         
        x = self.patch_conv(x)        
        x = self.act(self.bn(x))
        return x    




class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout,k1,k2,k3,p1,p2,p3):
        super(EnEmbedding, self).__init__()
        
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.non_overlappatch=ThreeLayerNonOverlappingPatchConv(n_vars,d_model)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, 30, d_model).normal_(std=0.02)
        ) 
       
        self.spectral_embedding=SpectralPatchMaxPool(d_model)
        self.multiscalepatch=MultiScalePatchEmbed(d_model)
        self.spectralpatch=SpectralPatch(d_model,50)
        self.dropout = nn.Dropout(dropout)


        self.embedding_20 = nn.Sequential(
 
            nn.Conv1d(in_channels=n_vars, out_channels=64,
                    kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),    


            nn.Conv1d(in_channels=64, out_channels=128,
                    kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),   

            nn.Conv1d(in_channels=128, out_channels=d_model,
                    kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=9)     
        )

        self.embedding_30=nn.Sequential(
            nn.Conv1d(in_channels=n_vars, out_channels=64,
                                kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  


            nn.Conv1d(in_channels=64, out_channels=128,
                                kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  


            nn.Conv1d(in_channels=128, out_channels=d_model,
                                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=13, stride=6))
        
        self.embedding_30_256 = nn.Sequential(
          
            nn.Conv1d(n_vars, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64), 
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),    

            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), 
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),    

         
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), 
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),     

            
            nn.Conv1d(256, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=17, stride=1)
        )


        self.embed_decreasing_kernel_128 = nn.Sequential(
            nn.Conv1d(n_vars, 64, kernel_size=k1, stride=1, padding=k1//2),
            nn.BatchNorm1d(64), 
            nn.GELU(),
            nn.MaxPool1d(p1,p1),            

            nn.Conv1d(64, 128, kernel_size=k2, stride=1, padding=k2//2),
            nn.BatchNorm1d(128), 
            nn.GELU(),
            nn.MaxPool1d(p2,p2),             

            nn.Conv1d(128, d_model, kernel_size=k3, stride=1, padding=k3//2),
            nn.BatchNorm1d(d_model), 
            nn.GELU(),
            nn.MaxPool1d(p3,p3),            
        )

        self.embed_decreasing_kernel_128_2layer = nn.Sequential(
            nn.Conv1d(n_vars, 128, kernel_size=k1, stride=1, padding=k1//2), 
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=p1, stride=p1),  

            nn.Conv1d(128, d_model, kernel_size=k2, stride=1, padding=k2//2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=p2, stride=p2), 
        )

        self.embed_decreasing_kernel_128_1layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=d_model, 
                kernel_size=k1,  
                stride=1,
                padding=50       
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=p1, stride=p1)  
        )
        
        self.embed_with_larger_stride = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=11, stride=10, padding=5),  
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=10, stride=10)  
        )

        self.embed1layer_nooverlap = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=100,  
                stride=100,        
                padding=0          
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU()
           
        )
        self.one_layer_cnn_nooverlap = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=5,   
                stride=5,        
                padding=2        
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=5)
        )
        self.one_layer_cnn_overlap = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=5,   
                stride=5,       
                padding=2        
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4) 
        )


        self.embed_decreasing_kernel_256 = nn.Sequential(
            nn.Conv1d(n_vars, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64), 
            nn.GELU(),
            nn.MaxPool1d(5,5),            

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128), 
            nn.GELU(),
            nn.MaxPool1d(4,4),             

            nn.Conv1d(128, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), 
            nn.GELU(),
            nn.MaxPool1d(5,5),

            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(1, 1)            
        )
        self.cnn1=nn.Conv1d(n_vars, 64, kernel_size=49, stride=6, bias=False, padding=24)
        self.cnn2=nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4)
        self.cnn3=nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4)

        self.pool1=nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        self.pool2=nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        self.pool3=nn.MaxPool1d(kernel_size=9, stride=2, padding=4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)


        self.act = nn.GELU()

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
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x=self.value_embedding(x)
        
        x=x.transpose(1,2)
   
        x = x + self.pos_embedding
        #x=apply_rotary_pos_emb(x)
        #x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
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
        # for layer in self.layers:
        #     x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        #last_self_attn = None
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
            #last_self_attn = self_attn

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
        #cross_normed=cross_seq # no prenorm
        cross_input = cross_normed  
        cross_self_out, eog_self_attn = self.self_attention(
            cross_input,   
            cross_input,    
            cross_input,   
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )

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

        eeg_self_out = eeg_self_out  
        x = x + self.dropout(eeg_self_out)          
        x = self.norm1(x)                            


        eeg_glb_ori = x[:, -1:, :]         
 
        eeg_glb_input = eeg_glb_ori 
        eog_glb_input = cross_glb_ori 

        # eeg_eog_global=torch.cat([eeg_glb_input,eog_glb_input],dim=1)
        # cross_out_eeg_eog, eeg_eog_cross_attn = self.cross_attention(
        #     eeg_eog_global,    
        #     eeg_eog_global,      
        #     eeg_eog_global,    
        #     attn_mask=None,
        #     tau=tau, delta=delta
        # )
        #print(eeg_eog_cross_attn.shape)
        cross_out_eeg, eeg_cross_attn = self.cross_attention(
            eeg_glb_input,    
            eog_glb_input,      
            eog_glb_input,    
            attn_mask=None,
            tau=tau, delta=delta
        )
 
        cross_out_eog, eog_cross_attn = self.cross_attention(
            eog_glb_input,    
            eeg_glb_input,     
            eeg_glb_input,    
            attn_mask=None,
            tau=tau, delta=delta
        )

        #eog_glb=cross_glb_ori

        eog_glb=cross_glb_ori +self.dropout(cross_out_eog)
        eog_glb=self.norm6(eog_glb)
        #cross_out = cross_out 

        #eeg_glb=eeg_glb_ori
        eeg_glb = eeg_glb_ori + self.dropout(cross_out_eeg)  
        eeg_glb = self.norm2(eeg_glb)
 

        x_patches_eeg = x[:, :-1, :]           


        #concatenate
        x_eeg = torch.cat([ x_patches_eeg,    
                        eeg_glb,        
                        #cross_glb_ori  
                      ], dim=1)

        x_patches_eog=cross_seq[:, :-1, :]
        x_eog = torch.cat([ x_patches_eog,    
                        eog_glb,       
                        #cross_glb_ori  
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

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(dim, dim * 2)
    
    def forward(self, x1, x2):
        gate = self.sigmoid(self.fc1(x1) + self.fc2(x2))
        fused = gate * x1 + (1 - gate) * x2
        return self.proj(fused)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.use_norm=True
        self.n_vars=1

        self.en_embedding_1 = EnEmbedding(self.n_vars, d_model=self.params.d_model, patch_len=self.params.patch_len, dropout=0.1,k1=self.params.k1,k2=self.params.k2,k3=self.params.k3,
                                          p1=self.params.p1,p2=self.params.p2,p3=self.params.p3)
        self.en_embedding_2 = EnEmbedding(self.n_vars, d_model=self.params.d_model, patch_len=self.params.patch_len, dropout=0.1,k1=self.params.k1,k2=self.params.k2,k3=self.params.k3,
                                          p1=self.params.p1,p2=self.params.p2,p3=self.params.p3)
        

        self.ex_embedding_1 = DataEmbedding_inverted(3000, d_model=self.params.d_model, embed_type='fixed', freq='s',
                                                   dropout=0.1)
        
        self.ex_embedding_2 = DataEmbedding_inverted(3000, d_model=self.params.d_model, embed_type='fixed', freq='s',
                                                   dropout=0.1)
        self.enc_embedding = DataEmbedding(c_in=1, d_model=self.params.d_model, embed_type='fixed', freq='s',
                                            dropout=0.1)
        
        
        self.token_embedding=DataEmbedding_wo_pos(c_in=1,d_model=self.params.d_model)
        self.patch_embedding=PatchEmbedding(d_model=self.params.d_model,patch_len=self.params.patch_len,stride=self.params.patch_len,padding=self.params.patch_len,dropout=self.params.dropout)
       
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


        self.fc_mu = nn.Linear(256, 256)
        self.fc_mu1 = nn.Linear(256,256)
        self.seq_encoder_global = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            depth=0,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
            return_attention=self.params.return_attention,
        )
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
        self.mlp1=MLPBlock(512, 512, 0.1)
        self.mlp2=MLPBlock(512,512, 0.1)
        self.gate1=GatedFusion(self.params.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)

        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.proj = nn.Linear(1024, 512)

        self.epoch_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(params.dropout),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )

        self.epoch_encoder2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(9, 2, padding=4), nn.Dropout(params.dropout),
            nn.Conv1d(64,128,9,1,4,bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(9,2,4),
            nn.Conv1d(128,256,9,1,4,bias=False), nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(9,2,4),
            nn.Conv1d(256,512,9,1,4,bias=False), nn.BatchNorm1d(512), nn.GELU(),
            nn.MaxPool1d(9,2,4),
        )

        self.gate_linear = nn.Linear(self.params.d_model * 2, 1)
        self.sigmoid = nn.Sigmoid()
        enc_layers_temporal = nn.TransformerEncoderLayer(
                            d_model=self.params.d_model,
                            nhead=self.params.num_heads,
                            dim_feedforward=2048,
                            dropout=self.params.dropout,
                            layer_norm_eps=1e-5,
                            batch_first=True, 
                            norm_first=True 
                        )
        
        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=self.params.d_model,
                                    nhead=self.params.num_heads,
                                    dim_feedforward=2048,
                                    dropout=self.params.dropout,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        self.gate_w1 = nn.Linear(self.params.d_model, self.params.d_model)
        self.gate_w2 = nn.Linear(self.params.d_model, self.params.d_model)
        self.gate_w3 = nn.Linear(self.params.d_model, self.params.d_model)
        self.gate_w4 = nn.Linear(self.params.d_model, self.params.d_model)
        self.gate_sigmoid = nn.Sigmoid()

        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=self.params.num_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=self.params.num_layers)

        self.head_nf = self.params.d_model * (self.params.patch_len + 1)
        self.head = FlattenHead(2, self.head_nf, self.params.d_model, head_dropout=self.params.dropout)
    

    
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
        mu         = self.fc_mu1(pooled_enc)   

        fused_glob = fused_glob.view(B, E, -1)
     
        glob_enc,_   = self.seq_encoder_global2(fused_glob)  

        fused_out  = self.fc_mu1(glob_enc)      

    
        return mu, fused_out, all_attns


        



        





