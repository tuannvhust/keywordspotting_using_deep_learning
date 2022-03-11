from sqlite3 import connect
from nnet.pooling import StatisticsPooling
from torch import nn
import torch.nn.functional as F
from .conv import DepthSeparableConv,DepthWiseConv
from .norm import SubSpectralNorm

class Xvector(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args.n_mels
        cnn_channels = [int(idx) for idx in args.cnn_channels.split(',')]
        cnn_kernel = [int(idx) for idx in args.cnn_kernel.split(',')]
        cnn_dilation = [int(idx) for idx in args.cnn_dilation.split(',')]
        n_embed = args.n_embed
        self.blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU()

        for block_index in range(len(cnn_channels)):
            out_channels = cnn_channels[block_index]
            self.blocks.extend([nn.Conv1d(in_channels=in_channels,
                                            out_channels = out_channels,
                                            kernel_size = cnn_kernel[block_index],
                                            dilation = cnn_dilation[block_index]),
                                self.activation,
                                nn.BatchNorm1d(out_channels)])
            in_channels = cnn_channels[block_index]
        
        self.pooling = StatisticsPooling()
        self.linear = nn.Sequential(nn.Linear(out_channels*2,n_embed),
                                    self.activation,
                                    nn.BatchNorm1d(n_embed),
                                    nn.Linear(n_embed,n_embed),
                                    self.activation,
                                    nn.BatchNorm1d(n_embed)
                                                )
    def forward(self,x):
        #x : N X T X F
        x = x.transpose(1,2)
        #x : N X F X T
        for layer in self.blocks:
            x = layer(x)
            #x : N X F X T
        x = x.transpose(1,2)
        # x : N X T X F
        x = self.pooling(x)
        # x : N x 1 x 2F
        x = x.squeeze(1)
        # x : N X 2F
        x = self.linear(x)
        # x : N X E
        return x 

class Xattention(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args.n_mels
        cnn_channels = [int(idx) for idx in args.cnn_channels.split(',')]
        cnn_kernel = [int(idx) for idx in args.cnn_kernel.split(',')]
        cnn_dilation = [int(idx) for idx in args.cnn_dilation.split(',')]
        n_heads = args.n_heads
        n_embed = args.n_embed
        self.blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU()

        for block_index in range(len(cnn_channels)):
            out_channels = cnn_channels[block_index]
            self.blocks.extend([nn.Conv1d(in_channels=in_channels,
                                            out_channels = out_channels,
                                            kernel_size = cnn_kernel[block_index],
                                            dilation = cnn_dilation[block_index]),
                                self.activation,
                                nn.BatchNorm1d(out_channels)])
            in_channels = cnn_channels[block_index]
        self.attention = nn.MultiheadAttention(embed_dim = out_channels,num_heads = n_heads)
        self.bn = nn.BatchNorm1d(out_channels)
        self.poolimg = StatisticsPooling()
        self.fc1 = nn.Sequential(nn.Linear(2*out_channels,n_embed),
                                self.activation,
                                nn.BatchNorm1d(n_embed))
        self.fc2 = nn.Sequential(nn.Linear(n_embed,n_embed),
                                self.activation,
                                nn.BatchNorm1d(n_embed))
    def forward(self,x):
        # x : N x T x F
        x = x.transpose(1,2)
        # x : N x F x T
        for layer in self.blocks:
            x = layer(x)
            # x: N x F x T
        x_p = x.permute(2,0,1)
        # x : T x N X F
        x_attention = self.attention(x_p,x_p,x_p)
        # x : T X N X F
        x_attention = x_attention.permute(1,2,0)
        # x : N X F X T
        x_attention = self.activation(x_attention)
        # x : N X F X T

        x = x + x_attention
        # x : N X F X T
        x = self.bn(x)
        # x : N X F X T
        x = x.transpose(1,2)
        # x : N X T X F
        x = self.pooling(x)
        # x : N x 1 x 2F
        x = x.squeeze(1)
        # x : N x 2F
        x = self.fc1(x)
        # x : N X E
        x = self.fc2()
        # x : N X E
        return x
class BCResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_5x5 = nn.Sequential(nn.Conv2d(1, 128, 5, stride=(2,1), padding=2),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())

        self.bcres_1 = BCResBlock(128, 64, 1, 1)
        self.bcres_2 = BCResBlock(64, 64, 1, 1)

        self.bcres_3 = BCResBlock(64, 64, (2,1), (1,2))
        self.bcres_4 = BCResBlock(64, 96, 1, (1,2))

        self.bcres_5 = BCResBlock(96, 96, (2,1), (1,4))
        self.bcres_6 = BCResBlock(96, 96, 1, (1,4))
        self.bcres_7 = BCResBlock(96, 96, 1, (1,4))
        self.bcres_8 = BCResBlock(96, 128, 1, (1,4))

        self.bcres_9 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_10 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_11 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_12 = BCResBlock(128, 160, 1, (1,8))

        self.dwconv_5x5 = nn.Conv2d(160, 160, 5, groups=20, padding=(0,2))
        self.conv_1x1 = nn.Conv2d(160, 256, 1)

        self.conv_out = nn.Conv2d(256, args.n_embed, 1)

    def forward(self, x):
        # input: N x T x F
        x = x.transpose(1,2).unsqueeze(1)
        # N x 1 x F x T

        x = self.conv_5x5(x)

        x = self.bcres_1(x)
        x = self.bcres_2(x)

        x = self.bcres_3(x)
        x = self.bcres_4(x)

        x = self.bcres_5(x)
        x = self.bcres_6(x)
        x = self.bcres_7(x)
        x = self.bcres_8(x)

        x = self.bcres_9(x)
        x = self.bcres_10(x)
        x = self.bcres_11(x)
        x = self.bcres_12(x)

        x = self.dwconv_5x5(x)
        x = self.conv_1x1(x)
        x = x.mean(-1).unsqueeze(-1)
        x = self.conv_out(x)
        x = torch.squeeze(x)

        return x


class BCResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1, S=5):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.F2 = nn.Sequential(nn.Conv2d(in_size, out_size, 1),
                                nn.BatchNorm2d(out_size),
                                nn.LeakyReLU(),
                                DepthWiseConv(out_size, out_size, stride, dilation),
                                SubSpectralNorm(out_size, S)
                               )
        self.F1 = nn.Sequential(DepthSeparableConv(out_size, out_size),
                                nn.BatchNorm2d(out_size),
                                nn.SiLU(),
                                nn.Conv2d(out_size, out_size, 1)      
                                )
    def forward(self, x):
        x2 = self.F2(x)
        xp = x2.mean(2).unsqueeze(2)
        x1 = self.F1(xp)
        return F.leaky_relu(x2 + x1)

    