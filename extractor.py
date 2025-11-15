import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, batchnorm = False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2))
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2))
    return layer

class Encoder(nn.Module):
    def __init__(self, output_dim=32, bn = None): 
        super(Encoder, self).__init__()
        self.bn = bn

        self.block0 = conv_block(1, 16, kernel_size = 3, stride = 1, padding=1, batchnorm = bn)
        self.block1 = conv_block(16, 32, kernel_size = 4, stride = 2, batchnorm = bn)
        self.block2 = conv_block(32, 32, kernel_size = 3, stride = 2, batchnorm = bn)
        self.block3 = conv_block(32, 32, kernel_size = 3, stride = 2, batchnorm = bn)
        self.block4 = conv_block(32, output_dim, kernel_size = 3, stride = 2, batchnorm = bn)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

            x_history = {}
            x = self.block0(x)
            x_history['1x'] = x

            x = self.block1(x)
            x_history['2x'] = x

            x = self.block2(x)
            x_history['4x'] = x

            x = self.block3(x)
            x_history['8x'] = x

            x = self.block4(x)
            x_history['16x'] = x

            fmap1s = {}
            fmap2s = {}
            for k in x_history:
                fmap1s[k] = torch.split(x_history[k], [batch_dim, batch_dim], dim=0)[0]
                fmap2s[k] = torch.split(x_history[k], [batch_dim, batch_dim], dim=0)[1]

            return fmap1s, fmap2s
        
        else:
            x = self.block4(self.block3(self.block2(self.block1(x))))
            return x
