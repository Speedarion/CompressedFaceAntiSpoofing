import torch
from torch import nn
from torchvision import models
import numpy as np

from collections import OrderedDict 

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
        dense_rgb = models.densenet161(pretrained=pretrained)
        dense_d = models.densenet161(pretrained=pretrained)
        features_rgb = list(dense_rgb.features.children())
        features_d = list(dense_d.features.children())
        temp_layer = features_d[0]
        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 96 filters
        new_weight = np.zeros((96,1,7,7))
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight
        features_d[0]=nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        features_d[0].weight.data = torch.Tensor(new_weight)
        self.enc_rgb = nn.Sequential(*features_rgb[0:8])
        self.enc_d = nn.Sequential(*features_d[0:8])
        self.linear=nn.Linear(768,1)
        self.linear_rgb=nn.Linear(384,1)
        self.linear_d=nn.Linear(384,1)
        self.gavg_pool=nn.AdaptiveAvgPool2d(1)


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
        gap_rgb = self.gavg_pool(enc_rgb).squeeze() 
        gap_d = self.gavg_pool(enc_d).squeeze() 
        gap_d=gap_d.view(-1,384)
        gap_rgb=gap_rgb.view(-1,384)
        gap_rgb = nn.Sigmoid()(gap_rgb) 
        gap_d = nn.Sigmoid()(gap_d) 
        
        #op_rgb and op_d are scores at respective classification heads(RGB and depth)
        op_rgb=self.linear_rgb(gap_rgb)
        op_d=self.linear_d(gap_d)
        op_rgb = nn.Sigmoid()(op_rgb)
        op_d = nn.Sigmoid()(op_d)

        #concatenation of embedding from both streams
        gap=torch.cat([gap_rgb,gap_d], dim=1)
        #op is score at joint head 
        op = self.linear(gap)
        op = nn.Sigmoid()(op)
 
        return gap, op, op_rgb, op_d


if __name__ == '__main__':

    """

    hooks = {}
    def forward_hook(layer_name):
        def hook(module, input, output):
            hooks[layer_name] = output
        return hook
    """
    input=torch.randn(32, 4,224,224, dtype=torch.float)
    #print(input[:, [0,1,2], :, :].shape)
    print(input[:,3,:,:].unsqueeze(1).shape)
    print(input[:, 3, :, :].unsqueeze(0).shape)
    #model=RGBDMH(pretrained=True,num_channels=1)
    model = models.densenet161(pretrained=True)
    num = 0
    for name, tensor in model.named_parameters():

        if len(tensor.shape)==4:
            print(name)
            num +=1
    print(num)

    """
    rgb1=model.enc_rgb[4].denselayer6.register_forward_hook(forward_hook('rgb1'))
    rgb2=model.enc_rgb[6].denselayer12.register_forward_hook(forward_hook('rgb2'))
    d1=model.enc_d[4].denselayer6.register_forward_hook(forward_hook('depth1'))
    d2=model.enc_d[6].denselayer12.register_forward_hook(forward_hook('depth2'))
    y=model(input)
    rgb1.remove()
    rgb2.remove()
    d1.remove()
    d2.remove()
    import pdb; pdb.set_trace()

    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[0])
    print(y[1])
    print(y[2])
    """

