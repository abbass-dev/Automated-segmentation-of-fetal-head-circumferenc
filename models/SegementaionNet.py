import torch 
import torch.nn as nn
import torch.nn.functional as F

class SegementaionNet(nn.Module):
    def __init__(self,params):
        super(SegementaionNet,self).__init__()
        c_in,h_in,w_in = params["input_shape"]
        init_f = params['initial_filter']
        num_out = params['num_outputs']
        #down
        self.conv1 = nn.Conv2d(c_in,init_f,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(init_f,2*init_f,kernel_size=3,padding=1,stride=1)
        self.conv3 = nn.Conv2d(2*init_f,4*init_f,kernel_size=3,padding=1,stride=1)
        self.conv4 = nn.Conv2d(4*init_f,8*init_f,kernel_size=3,padding=1,stride=1)
        self.conv5 = nn.Conv2d(8*init_f,16*init_f,kernel_size=3,padding=1,stride=1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        #up
        self.convup1 = nn.Conv2d(16*init_f,8*init_f,kernel_size=3,padding=1,stride=1)
        self.convup2 = nn.Conv2d(8*init_f,4*init_f,kernel_size=3,padding=1,stride=1)
        self.convup3 = nn.Conv2d(4*init_f,2*init_f,kernel_size=3,padding=1,stride=1)
        self.convup4 = nn.Conv2d(2*init_f,init_f,kernel_size=3,padding=1,stride=1)
        self.convup5 = nn.Conv2d(init_f,c_in,kernel_size=3,padding=1,stride=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv5(x))
        
        x = self.upsample(x)
        x = F.relu(self.convup1(x))
        x = self.upsample(x)
        x = F.relu(self.convup2(x))
        x = self.upsample(x)
        x = F.relu(self.convup3(x))
        x = self.upsample(x)
        x = F.relu(self.convup4(x))
        x = self.convup5(x)
        return x
        
