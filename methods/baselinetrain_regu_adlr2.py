from methods import backbone#_resNest_combine_all_rfcov_shuffle_group2_swish_pycov_3covn   #
from methods.regularization import Regularization
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import xlsxwriter
# --- conventional supervised training ---

#workbook_loss_encoder = xlsxwriter.Workbook('/Extra/qifan_data/few-shot/output/Extra/checkpoints/acc_one_encoder.xlsx')
#worksheet_loss_encoder = workbook_loss_encoder.add_worksheet()

     
class BaselineTrain(nn.Module):
  def __init__(self, model_func, num_class, tf_path=None, loss_type = 'softmax'):
    super(BaselineTrain, self).__init__()

    # feature encoder
    self.feature    = model_func()
    # self.model = model
    # loss function: use 'dist' to pre-train the encoder for matchingnet, and 'softmax' for others
    if loss_type == 'softmax':
      self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
      self.classifier.bias.data.fill_(0)
    elif loss_type == 'dist':
      self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
    self.loss_type = loss_type
    self.loss_fn = nn.CrossEntropyLoss()

    self.num_class = num_class
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  def forward(self,x):
    x = x.cuda()
    out  = self.feature.forward(x)
    scores  = self.classifier.forward(out)
    return scores

  def forward_loss(self, x, y):
    scores = self.forward(x)
    y = y.cuda()
    return self.loss_fn(scores, y )

  def train_loop(self, epoch, train_loader, optimizer, total_it):#

    print_freq = len(train_loader) // 10
    avg_loss=0
    
    for i, (x,y) in enumerate(train_loader):
      optimizer.zero_grad()
           
      loss = self.forward_loss(x, y)


      loss.backward()
      #scheduler.step(loss)# 
      optimizer.step()
      lr = optimizer.param_groups[0]['lr']#
      avg_loss = avg_loss+loss.item()#data[0]

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)  ))
        #worksheet_loss_encoder.write(epoch+1, 0, avg_loss/float(i+1)) 
        #file_handle.write( ' \n ' + str(avg_loss/float(i+1)) )
        #with open('/Extra/qifan_data/few-shot/output/Extra/checkpoints/loss_gnnlayer3_attn.txt','a') as f:
        #    f.write(' \n ' + str(avg_loss/float(i+1)))
        print(epoch, lr)#
      if (total_it + 1) % 10 == 0:
        self.tf_writer.add_scalar('loss', loss.item(), total_it + 1)
      total_it += 1
    #f.close()
    return total_it

  def test_loop(self, val_loader):
    return -1 #no validation, just save model during iteration

