import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sequence_loss(img_preds, flow_preds, fixed, loss_fn, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_sim_loss = 0.0
    flow_grad_loss = 0.0

    for i in range(n_predictions):
        y_pred = img_preds[i]
        i_weight = gamma**(n_predictions - i - 1)
        i_sim_loss, i_grad_loss = loss_fn.loss(fixed, y_pred, flow_preds[i])
        flow_sim_loss += i_weight * (i_sim_loss).mean()
        flow_grad_loss += i_weight * (i_grad_loss).mean()

    return flow_sim_loss, flow_grad_loss


class Loss_Fn:
    def __init__(self, fname='mse', penalty='l1', weight=0.01, device='cuda:0'):
        if fname == 'mse':
            self.fn1 = MSE()
        elif fname == 'ncc':
            self.fn1 = NCC()

        self.fn2 = Grad3d(penalty)
        self.weight = weight
        
    def loss(self, y_true, y_pred, flow):
        return self.fn1.loss(y_true, y_pred), self.weight * self.fn2.loss(flow)


class MSE():
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class NCC():
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def loss(self, y_pred, y_true):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)



class Grad3d():
    
    """  N-D gradient loss. """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
