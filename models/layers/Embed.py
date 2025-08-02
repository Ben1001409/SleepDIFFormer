import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(x.shape)
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        #x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x).transpose(1, 2)
        #print(x.shape)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x=x.permute(0,2,1)
        #print(x.shape)
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x.permute(0,2,1))
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class ExogTokenCNN(nn.Module):

    def __init__(self, in_channels=1, d_model=128):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,  
            padding_mode='circular',
            bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1) 

        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        #print(x.shape)
        x=x.transpose(1,2)
        x = self.conv(x)           
        x = self.bn(x)
        x = self.act(x)
        #print(x.shape)
        x = self.pool(x)           
        x = x.transpose(1, 2)      
        return x

class TwoLayerConv(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model // 2,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.act1 = nn.GELU()


        self.conv2 = nn.Conv1d(
            in_channels=d_model // 2,
            out_channels=d_model,
            kernel_size=3,
            stride=50, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):  
        x = self.conv1(x)   
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)   
        x = self.bn2(x)
        x = self.act2(x)
        x=self.pool(x)
        return x.permute(0, 2, 1)  



class ConvToSingleToken(nn.Module):
    def __init__(self, in_channels=1, d_model=128):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=49,
            stride=6,
            padding=24,
            bias=False
        )
        self.bn1   = nn.BatchNorm1d(64)
        self.act1  = nn.GELU()
        self.pool1 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)


        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False
        )
        self.bn2   = nn.BatchNorm1d(128)
        self.act2  = nn.GELU()
        self.pool2 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
  

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=d_model,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False
        )
        self.bn3   = nn.BatchNorm1d(d_model)
        self.act3  = nn.GELU()
        self.pool3 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  

    def forward(self, x):  
        x=x.permute(0,2,1)
        x = self.act1(self.bn1(self.conv1(x)))  
        x = self.pool1(x)                      

        x = self.act2(self.bn2(self.conv2(x)))  
        x = self.pool2(x)                      

        x = self.act3(self.bn3(self.conv3(x))) 
        x = self.pool3(x)                      
        x=x.permute(0,2,1)
        #print(x.shape)
        #x = self.pool(x)                       
        return x
                 
#cnn as value embedding
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.positional_encoding=PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.cnn=ExogTokenCNN(in_channels=c_in,d_model=d_model)
        self.cnn1=nn.Conv1d(in_channels=c_in,out_channels=64,kernel_size=5,stride=2,padding=2)
        self.cnn2=nn.Conv1d(in_channels=64,out_channels=d_model,kernel_size=25,stride=25,padding=0)
        self.act1=nn.GELU()
        self.act2=nn.GELU()
        self.conv2layer=TwoLayerConv(c_in,d_model)
        self.conv3layer=ConvToSingleToken(c_in,d_model)
    def forward(self, x, x_mark):
        #x = x.permute(0, 2, 1)
        #print("DEI shape ",x.shape)
        # x: [Batch Variate Time]
        if x_mark is None:
 
            x=self.conv3layer(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
            #print(x.shape)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars