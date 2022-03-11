import torch
import torch.nn.functional as F
import torch.nn as nn

# blocks
def conv_bn_act(in_size, out_size, kernel_size, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_size, out_size, kernel_size, stride, dilation=dilation),
        nn.BatchNorm1d(out_size),
        nn.LeakyReLU()
    )

def sepconv_bn(in_size, out_size, kernel_size, stride=1, dilation=1, padding=None):
    if padding is None:
        padding = (kernel_size-1)//2
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size, 
                        stride=stride, dilation=dilation, groups=in_size,
                        padding=padding),
        torch.nn.Conv1d(in_size, out_size, kernel_size=1),
        nn.BatchNorm1d(out_size)
    )

# Main block B_i
class QnetBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1,
                R=5):
        super().__init__()

        self.layers = nn.ModuleList(sepconv_bn(in_size, out_size, kernel_size, stride))
        for i in range(R - 1):
            self.layers.append(nn.LeakyReLU())
            self.layers.append(sepconv_bn(out_size, out_size, kernel_size, stride))
        self.layers = nn.Sequential(*self.layers)

        self.residual = nn.ModuleList()
        self.residual.append(torch.nn.Conv1d(in_size, out_size, kernel_size=1))
        self.residual.append(torch.nn.BatchNorm1d(out_size))
        self.residual = nn.Sequential(*self.residual)

    def forward(self, x):
        return F.silu(self.residual(x) + self.layers(x))

class QuartzNet(nn.Module):
    def __init__(self, n_mels, num_classes):
        super().__init__()
        self.c1 = sepconv_bn(n_mels, 64, kernel_size=3, stride=2)
        self.blocks = nn.Sequential(
                #         in   out  k   s  R
                QnetBlock(64, 64, 5, 1, R=5),
                QnetBlock(64, 64, 3, 1, R=5),
                QnetBlock(64, 64, 3, 1, R=5),
                QnetBlock(64, 64, 1, 1, R=5),
                QnetBlock(64, 64, 1, 1, R=5)
        )
        self.c2 = sepconv_bn(64, 64, kernel_size=3, dilation=2, padding=1)
        self.c3 = conv_bn_act(64, 64, kernel_size=1)
        self.c4 = conv_bn_act(64, num_classes, kernel_size=1)
        self.attn = nn.MultiheadAttention(num_classes, num_heads=4)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        c1 = F.relu(self.c1(x))
        blocks = self.blocks(c1)
        c2 = F.relu(self.c2(blocks))
        c3 = self.c3(c2)
        c4 = self.c4(c3) #N x F x T
        c5 = c4.permute(2, 0, 1) #T x N x F
        c5, _ = self.attn(c5, c5, c5)
        c5 = c5.permute(1, 2, 0)
        c5 = self.bn(c5)
        return (c5+c4).mean(-1)

if __name__ == '__main__':
    model = QuartzNet(40, 512)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(4, 40, 101)
    y = model(x)
    print(y.shape)