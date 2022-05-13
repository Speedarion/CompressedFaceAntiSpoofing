import torch
from torch import nn
from torchvision import models
import numpy as np
import sys
sys.path.insert(0,'/home/hazeeq/FYP-Hazeeq/models/pytorch_image_models')
from timm.models.ghostnet import ghostnet_100,ghostnet_130

class RGBDMH(nn.Module):

    """ 

    Two-stream RGBD architecture

    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True 
    num_channels: int
        Number of channels in the input.      
    """

    def __init__(self, pretrained=True, num_channels=4):

        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        num_channels: int
            Number of channels in the input. 
        """
        super(RGBDMH, self).__init__()

        ghost_rgb = ghostnet_130(pretrained=pretrained)

        ghost_d = ghostnet_130(pretrained=pretrained)
        #import pdb; pdb.set_trace()

        ghost_rgb.classifier = nn.Linear(1280,1,bias=True)
        ghost_d.classifier = nn.Linear(1280,1,bias=True)


        features_rgb = list(ghost_rgb.children())

        features_d = list(ghost_d.children())

        #import pdb; pdb.set_trace()


        temp_layer = features_d[0]

        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 16 filters

        new_weight = np.zeros((20,1,3,3))
  
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight

        features_d[0]=nn.Conv2d(1, 20, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        features_d[0].weight.data = torch.Tensor(new_weight)

        self.enc_rgb = nn.Sequential(*features_rgb[0:8])

        self.enc_d = nn.Sequential(*features_d[0:8])

        self.linear=nn.Linear(2560,1)

        self.linear_rgb=nn.Linear(1280,1)

        self.linear_d=nn.Linear(1280,1)
        #import pdb; pdb.set_trace()




    def forward(self, img):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects Multi-channel images of size num_channelsx224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor` 
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.  
        gap: Gobal averaged pooling from the encoded feature maps

        """

        x_rgb = img[:, [0,1,2], :, :]

        x_depth = img[:, 3, :, :].unsqueeze(1)

        enc_rgb = self.enc_rgb(x_rgb)

        enc_d = self.enc_d(x_depth)
        """
        gap_rgb = self.gavg_pool(enc_rgb).squeeze() 
        gap_d = self.gavg_pool(enc_d).squeeze() 

        gap_d=gap_d.view(-1,960)

        gap_rgb=gap_rgb.view(-1,960)
        
        gap_rgb = nn.Sigmoid()(gap_rgb) 
        gap_d = nn.Sigmoid()(gap_d) 
        """
        op_rgb=self.linear_rgb(enc_rgb)

        op_d=self.linear_d(enc_d)
        #import pdb; pdb.set_trace()

        op_rgb = nn.Sigmoid()(op_rgb)

        op_d = nn.Sigmoid()(op_d)
        #import pdb; pdb.set_trace()

        joint=torch.cat([enc_rgb,enc_d], dim=1)
        op = self.linear(joint)
        op = nn.Sigmoid()(op)
 
        return  op, op_rgb, op_d

"""

model = RGBDMH(pretrained=True,num_channels=4)
input=torch.randn(32, 4,224,224, dtype=torch.float)
y=model(input)
import pdb; pdb.set_trace()


input=torch.randn(32, 4,224,224, dtype=torch.float)
#print(input[:, [0,1,2], :, :].shape)
print(input[:,3,:,:].unsqueeze(1).shape)
print(input[:, 3, :, :].unsqueeze(0).shape)
model=RGBDMH(pretrained=True,num_channels=1)
y=model(input)
print(y[0].shape)
print(y[1].shape)
print(y[2].shape)
print(y[0])
print(y[1])
print(y[2])
"""

