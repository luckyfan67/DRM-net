import torch
import torch.nn as nn
import numpy as np
from methods.meta_template_mutual import MetaTemplate
from methods.gnn_q8_elu_mutual_5shot1 import GNN_nl
from methods.q8gnn_5shot1 import *
from methods import backbone12
from methods.seevodes1 import *
class GnnNet_m(MetaTemplate):
  maml=True
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(GnnNet_m, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone12.Linear_fw(self.feat_dim, 128), backbone12.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn_m = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn_m.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)#15
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores,mu_loss = self.forward_gnn(z_stack)
    return scores,mu_loss

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    #query set
    quy_nodes = torch.cat([nodes[:,6*(i+1)-1:6*(i+1),:] for i in range(5)],dim=1)
    #suppport set
    spt_nodes = torch.cat([nodes[:,6*i:6*i+5,:] for i in range(5)],dim=1)
    spt_nodes_nolab = spt_nodes[:,:,0:128]
    spt_nodes_nolab1 =  spt_nodes_nolab[0:1,:,:].view(-1,128)
    #spt_nodes_nolab1 = spt_nodes[0:1, :, :].view(-1, 133)
    protolab1 = torch.chunk(spt_nodes_nolab1,5,dim=0)
    #expeand to 10 shot
    spt_exp_lab = []
    for i in range(5):

      for j in range(5):
        tmp = protolab1[i]
        tmp = tmp[torch.arange(tmp.size(0))!=j]
        tmp1 = torch.mean(tmp, dim=0).view(-1, 1, 128)
        spt_exp_lab.append(tmp1)
    
    spt_exp = torch.cat([spt_exp_lab[i] for i in range(len(spt_exp_lab))],dim=1)
    spt_exp_batch = spt_exp.repeat_interleave(nodes.shape[0], 0)
    spt_exp_batch_lab = torch.chunk(spt_exp_batch, 5, dim=1)
    
    protolab = torch.chunk(spt_nodes_nolab,5,dim=1) ##
    
    combine_exp_proto = torch.cat([torch.cat([protolab[i],spt_exp_batch_lab[i]],dim=1) for i in range(5)],dim=1)
    """
    allproto1 = torch.cat([torch.mean(protolab1[i],dim=0).view(-1,1,128) for i in range(len(protolab1))],dim=1)

    allproto_batch = allproto1.repeat_interleave(16,0) ##
    spt_lab = spt_nodes[:,:,128:]
    
    catlablst = []
    for i in range(len(protolab)):
      all_spt = torch.cat([protolab[i],allproto_batch[:,i:i+1,:]], dim=1)
      catlablst.append(all_spt)
    cat_spt = torch.cat([catlablst[i] for i in range(len(catlablst))],dim=1)
    """
    support_label_new = torch.from_numpy(np.repeat(range(self.n_way), self.n_support+5)).unsqueeze(1)
    support_label_new = torch.zeros(self.n_way * (self.n_support+5), self.n_way).scatter(1, support_label_new, 1).view(self.n_way,self.n_support+5,self.n_way)
    #support_label_new = torch.cat([support_label_new, torch.zeros(self.n_way, 1, 5)], dim=1)
    support_label_all = support_label_new.view(-1,5).view(-1,50,5).repeat_interleave(nodes.shape[0],0)
    spt_all_nodes = torch.cat([combine_exp_proto.cuda(),support_label_all.cuda()],dim=2)
    

    all_lst =[]
    for i in range(5):

      combine_sq = torch.cat([spt_all_nodes[:, 10*i:10*i + 10, :], quy_nodes[:, i:i + 1, :]],dim=1)
      #combine_sq = torch.cat([spt_nodes[:, i:i + 1, :], spt_zero_nodes[:, i:i + 1, :], quy_nodes[:, i:i + 1, :]], dim=1)
      all_lst.append(combine_sq)  
    #combine support and query
    all_nodes = torch.cat([all_lst[i] for i in range(len(all_lst))],dim=1)
    
    scores,mu_loss = self.gnn_m(all_nodes)
    #scores, mu_loss = self.gnn_m(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 6, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way) #2shot support+2 10shot+6

    return scores,mu_loss

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores,mu_loss = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss, mu_loss
