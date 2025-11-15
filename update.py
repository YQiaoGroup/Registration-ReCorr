import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, 3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv1.weight.shape))
        self.conv1.bias = nn.Parameter(torch.zeros(self.conv1.bias.shape))
        self.conv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv2.weight.shape))
        self.conv2.bias = nn.Parameter(torch.zeros(self.conv2.bias.shape))

    def forward(self, x):
        return self.conv2(self.leaky_relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convr1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convq1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))

        self.convz2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convr2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convq2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))

        self.convz3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))
        self.convr3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))
        self.convq3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        # spatial
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q
        return h    

    
class MotionEncoder(nn.Module):
    def __init__(self, corr_radius=3):
        super(MotionEncoder, self).__init__()
        cor_planes = (corr_radius)**3
        self.convc1 = nn.Conv3d(cor_planes, 16, 1, padding=0)
        self.convc2 = nn.Conv3d(16, 32, 3, padding=1)
        self.convf1 = nn.Conv3d(3, 16, 7, padding=3)
        self.convf2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv = nn.Conv3d(32+32, 32-3, 3, padding=1) # 128 → context_dim

    def forward(self, flow, corr):
        cor = F.leaky_relu(self.convc1(corr))
        cor = F.leaky_relu(self.convc2(cor))
        flo = F.leaky_relu(self.convf1(flow))
        flo = F.leaky_relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.leaky_relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        ''' hidden_dim: hidden dim
            input_dim: context_net outdim '''
        super(UpdateBlock, self).__init__()
        self.encoder = MotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=32+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=16)


    def forward(self, hidden, context, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([context, motion_features], dim=1)

        hidden = self.gru(hidden, inp)
        delta_flow = self.flow_head(hidden)

        return hidden, delta_flow

    


