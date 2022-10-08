# This code is modified from https://github.com/vgsatorras/few-shot-gnn/blob/master/models/gnn_iclr.py
import random
from random import uniform, randint
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from methods.backbone12 import Linear_fw, Conv2d_fw, BatchNorm2d_fw, BatchNorm1d_fw
from core_qnn.quaternion_layers import QuaternionLinear, QuaternionConv
from methods.util import DistillKL
from methods.q8gnn_5shot1 import *
from methods.seevodes import *

if torch.cuda.is_available():
  dtype = torch.cuda.FloatTensor
  dtype_l = torch.cuda.LongTensor
else:
  dtype = torch.FloatTensor
  dtype_l = torch.cuda.LongTensor

def gmul(input):
  W, x = input
  # x is a tensor of size (bs, N, num_features)
  # W is a tensor of size (bs, N, N, J)
  #x_size = x.size()
  W_size = W.size()
  N = W_size[-2]
  W = W.split(1, 3)
  W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
  output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
  output = output.split(N, 1)
  output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
  return output

def relu_fn(x):
 
    """ Swish activation function """
    return x * torch.sigmoid(x)

#se block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

class Gconv(nn.Module):
  maml = True
  def __init__(self, nf_input, nf_output, J, bn_bool=True):
    super(Gconv, self).__init__()
    self.J = J
    self.num_inputs = J*nf_input
    self.num_outputs = nf_output
    self.fc = nn.Linear(self.num_inputs, self.num_outputs) if not self.maml else Linear_fw(self.num_inputs, self.num_outputs)

    self.bn_bool = bn_bool
    if self.bn_bool:
      self.bn = nn.BatchNorm1d(self.num_outputs, track_running_stats=False) if not self.maml else BatchNorm1d_fw(self.num_outputs, track_running_stats=False)

  def forward(self, input):
    W = input[0]
    x = gmul(input) # out has size (bs, N, num_inputs)[16,30,266]
    #if self.J == 1:
    #    x = torch.abs(x)
    x_size = x.size()
    x = x.contiguous()
    x = x.view(-1, self.num_inputs)       #[480,266]
    x = self.fc(x) # has size (bs*N, num_outputs)  [480,48]

    if self.bn_bool:
      x = self.bn(x)
    x = x.view(*x_size[:-1], self.num_outputs)   #x[16,30,48]
    return W, x

class Wcompute(nn.Module):
  maml = True
  def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
    super(Wcompute, self).__init__()
    self.num_features = nf
    self.operator = operator
    #self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1) if not self.maml else Conv2d_fw(input_features, int(nf * ratio[0]), 1, stride=1)
    self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1) if not self.maml else Conv2d_fw(input_features, int(nf * ratio[0]), 1, stride=1)
    self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]), track_running_stats=False) if not self.maml else BatchNorm2d_fw(int(nf * ratio[0]), track_running_stats=False)
    #self.seblock = SELayer(input_features)#
    self.drop = drop
    #if self.drop:
    #  self.dropout = nn.Dropout(0.3)
    
    if self.drop:
      r = 0.1 * randint(1, 3)
      print(r)
      self.dropout = nn.Dropout(r) #0.3

    self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1) if not self.maml else Conv2d_fw(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
    self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]), track_running_stats=False) if not self.maml else BatchNorm2d_fw(int(nf * ratio[1]), track_running_stats=False)
    self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1) if not self.maml else Conv2d_fw(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
    self.bn_3 = nn.BatchNorm2d(nf*ratio[2], track_running_stats=False) if not self.maml else BatchNorm2d_fw(nf*ratio[2], track_running_stats=False)
    self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1) if not self.maml else Conv2d_fw(nf*ratio[2], nf*ratio[3], 1, stride=1)
    self.bn_4 = nn.BatchNorm2d(nf*ratio[3], track_running_stats=False) if not self.maml else BatchNorm2d_fw(nf*ratio[3], track_running_stats=False)
    self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1) if not self.maml else Conv2d_fw(nf, num_operators, 1, stride=1)
    self.activation = activation
    self.conv2d_04 = Conv2d_fw(input_features, nf, 1,stride=1)
    self.bn_04 = BatchNorm2d_fw(nf,track_running_stats=False)




  def forward(self, x, W_id):
    W1 = x.unsqueeze(2)
    W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
    W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
    W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N
    #W_new = self.seblock(W_new)

    W_new04 = self.conv2d_04(W_new) #
    W_new04 = self.bn_04(W_new04) #
    W_new04 = F.leaky_relu(W_new04) #


    
    W_new = self.conv2d_1(W_new)
    W_new = self.bn_1(W_new)
    W_new = F.elu(W_new)
    #if self.drop:
    #  W_new = self.dropout(W_new)

    if self.drop:
      r = 0.1 * randint(1, 3)
      #print(r)
      self.dropout = nn.Dropout(r)
      W_new = self.dropout(W_new)
  
    W_new = self.conv2d_2(W_new)
    W_new = self.bn_2(W_new)
    W_new = F.elu(W_new)

    W_new = self.conv2d_3(W_new)
    W_new = self.bn_3(W_new)
    W_new = F.elu(W_new)

    W_new = self.conv2d_4(W_new)
    W_new = self.bn_4(W_new)
    W_new = F.elu(W_new)

    W_new = W_new + W_new04 #

    W_new = self.conv2d_last(W_new)
    W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

    if self.activation == 'softmax':
      W_new = W_new - W_id.expand_as(W_new) * 1e8
      W_new = torch.transpose(W_new, 2, 3)
      # Applying Softmax
      W_new = W_new.contiguous()
      W_new_size = W_new.size()
      W_new = W_new.view(-1, W_new.size(3))
      W_new = F.softmax(W_new, dim=1)
      W_new = W_new.view(W_new_size)
      # Softmax applied
      W_new = torch.transpose(W_new, 2, 3)

    elif self.activation == 'sigmoid':
      W_new = F.sigmoid(W_new)
      W_new *= (1 - W_id)
    elif self.activation == 'none':
      W_new *= (1 - W_id)
    else:
      raise (NotImplementedError)

    if self.operator == 'laplace':
      W_new = W_id - W_new
    elif self.operator == 'J2':
      W_new = torch.cat([W_id, W_new], 3)
    else:
      raise(NotImplementedError)

    return W_new

class GNN_nl(nn.Module):
  def __init__(self, input_features, nf, train_N_way):
    super(GNN_nl, self).__init__()
    self.input_features = input_features
    self.nf = nf
    self.num_layers = 3 #3
    self.bn_1 = BatchNorm2d_fw(int(input_features), track_running_stats=False)
    self.mutual_loss = DistillKL(T=4)

    for i in range(self.num_layers):
      if i == 0:
        module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_lori = Gconv(self.input_features, int(nf / 2), 2)
        module_l = Q8GNNLayer(self.input_features*2, int(nf / 2), dropout=0.5, quaternion_ff=True, act=True)
      else:
        module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_lori = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
        module_l = Q8GNNLayer((self.input_features + int(nf / 2) * i)*2, int(nf / 2), dropout=0.5, quaternion_ff=True, act=True)
      self.add_module('layer_w{}'.format(i), module_w)
      self.add_module('layer_l{}'.format(i), module_l)
      self.add_module('layer_lori{}'.format(i), module_lori)

    self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
    self.layer_last_ori = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, train_N_way, 2, bn_bool=False)
    self.layer_last = Q8GNNLayer((self.input_features + int(self.nf / 2) * self.num_layers)*2, train_N_way, dropout=0.5, quaternion_ff=False, act=False)

  def forward(self, x):
    W_init = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)
    x_ori = x.clone()
    mutual_loss_all = 0
    for i in range(self.num_layers):
      Wi = self._modules['layer_w{}'.format(i)](x, W_init)
      x_new = F.elu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
      #x_new_ori = F.elu(self._modules['layer_lori{}'.format(i)]([Wi, x])[1])
      x = torch.cat([x, x_new], 2)
      #x_ori = torch.cat([x, x_new_ori], 2)


    #for i in range(self.num_layers):
      #Wi_ori = self._modules['layer_w{}'.format(i)](x_ori, W_init)
      x_new_ori = F.elu(self._modules['layer_lori{}'.format(i)]([Wi, x_ori])[1])
      x_ori = torch.cat([x_ori, x_new_ori], 2)
      mutual_loss1 = self.mutual_loss(x,x_ori)+self.mutual_loss(x_ori,x)
      mutual_loss_all = mutual_loss_all + mutual_loss1

    Wl=self.w_comp_last(x, W_init)
    Wl_ori = self.w_comp_last(x_ori, W_init)
    #out = self.layer_last([Wl, x])[1]   #[16,30,5]
    out = self.layer_last([Wl, x])[1] 
    out_ori = self.layer_last_ori([Wl_ori, x_ori])[1]
    
    out = out.view(x.size()[0], -1, 5)    #x_test[15,30,229],x_train[16,30,229]
    out_ori = out_ori.view(x_ori.size()[0], -1, 5)

    co_out = out+out_ori
    #tsne_drow(out)
    return co_out, mutual_loss_all
