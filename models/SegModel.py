from abc import ABC,abstractmethod
import torch
from models.SegementaionNet import SegementaionNet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models.base import Base
import torch.nn.functional as F
from losses import diceLoss

class SegModel(Base):
    def __init__(self,params):
        super(SegModel,self).__init__()
        self.network_names =[]
        self.loss_names= []
        self.optimizers = []
        self.netSeg = SegementaionNet(params).to(self.device)
        self.network_names.append('Seg')
        self.opt = Adam(self.netSeg.parameters(),lr=0.001)
        self.optimizers.append(self.opt)
        self.schedulers.append(StepLR(self.opt,step_size=10, gamma=0.3))
  
    def forward(self):
        self.output = self.netSeg(self.input)

    def backward(self):
        self.loss_bce = F.binary_cross_entropy_with_logits(self.output,self.label,reduction='sum')
        self.loss_names.append('bce')

        self.loss_names.append('dice')
        self.pred = torch.sigmoid(self.output)
        self.loss_dice,self.dice = diceLoss(self.pred,self.label)

        self.loss_names.append('final')
        self.loss_final = self.loss_bce+self.loss_dice

        

        #self.loss_dice = diceLoss(self.output,self.label)

    def optimize_parameters(self):
        self.loss_final.backward()
        self.opt.step()
        self.opt.zero_grad()
       
