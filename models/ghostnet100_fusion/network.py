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

        ghost_rgb = ghostnet_100(pretrained=pretrained)

        ghost_d = ghostnet_100(pretrained=pretrained)
        #import pdb; pdb.set_trace()

        ghost_rgb.classifier = nn.Linear(1280,1,bias=True)
        ghost_d.classifier = nn.Linear(1280,1,bias=True)


        features_rgb = list(ghost_rgb.children())

        features_d = list(ghost_d.children())

        #import pdb; pdb.set_trace()


        temp_layer = features_d[0]

        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 16 filters

        new_weight = np.zeros((16,1,3,3))
  
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight

        features_d[0]=nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        features_d[0].weight.data = torch.Tensor(new_weight)

        #features_rgb and features_d from index 0 to 8 excludes the original classifier layer in ghostnet
        self.enc_rgb = nn.Sequential(*features_rgb[0:8])

        self.enc_d = nn.Sequential(*features_d[0:8])

        self.linear=nn.Linear(2560,1)

        self.linear_rgb=nn.Linear(1280,1)

        self.linear_d=nn.Linear(1280,1)

        self.hooks = {}
        def forward_hook(layer_name):
            def hook(module, input, output):
                self.hooks[layer_name] = output.clone().detach()
            return hook
        self.gavg_pool=nn.AdaptiveAvgPool2d(1)

        #RGB Stream - 4th , 8th , 12th , 16th Ghost Bottleneck
        self.enc_rgb[3][3][0].ghost2.cheap_operation.register_forward_hook(forward_hook('rgb1'))
        self.enc_rgb[3][6][1].ghost2.cheap_operation.register_forward_hook(forward_hook('rgb2'))
        self.enc_rgb[3][7][0].ghost2.cheap_operation.register_forward_hook(forward_hook('rgb3'))
        self.enc_rgb[3][8][3].ghost2.cheap_operation.register_forward_hook(forward_hook('rgb4'))
        #Depth Stream - 4th , 8th , 12th , 16th Ghost Bottleneck
        self.enc_d[3][3][0].ghost2.cheap_operation.register_forward_hook(forward_hook('depth1'))
        self.enc_d[3][6][1].ghost2.cheap_operation.register_forward_hook(forward_hook('depth2'))
        self.enc_d[3][7][0].ghost2.cheap_operation.register_forward_hook(forward_hook('depth3'))
        self.enc_d[3][8][3].ghost2.cheap_operation.register_forward_hook(forward_hook('depth4'))

        # FUSION A - WEIGHTED SUM
        self.fusefc1 = nn.Linear(2,1)
        self.fusefc2 = nn.Linear(3,1)
        self.fusefc3 = nn.Linear(3,1)
        self.fusefc4 = nn.Linear(3,1)

        self.finalfc = nn.Linear(80,1)

        self.Relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(20,80,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(80)
        self.conv2 = nn.Conv2d(80,40,kernel_size=(3,3),stride=(1,1),bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.avgpool1 = nn.AdaptiveAvgPool2d(14)
        self.conv3 = nn.Conv2d(40,160,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn3 = nn.BatchNorm2d(160)
        self.conv4 = nn.Conv2d(160,80,kernel_size=(3,3),stride=(1,1),bias=False)
        self.bn4 = nn.BatchNorm2d(80)
        self.avgpool2 = nn.AdaptiveAvgPool2d(7)
        self.conv5 = nn.Conv2d(80,160,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn5 = nn.BatchNorm2d(160)
        self.conv6 = nn.Conv2d(160,80,kernel_size=(3,3),stride=(1,1),bias=False)
        self.bn6 = nn.BatchNorm2d(80)
        self.conv7 = nn.Conv2d(80,160,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn7 = nn.BatchNorm2d(160)
        self.conv8 = nn.Conv2d(160,80,kernel_size=(3,3),stride=(1,1),bias=False)
        self.bn8 = nn.BatchNorm2d(80)

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

        joint=torch.cat([enc_rgb,enc_d], dim=1)
        op = self.linear(joint)
        op = nn.Sigmoid()(op)

        #FUSION - A (weighted sum)
        #Stack feature maps from both streams along a new dimension
        input1=torch.stack([self.hooks['rgb1'],self.hooks['depth1']],dim=-1)
        #Fully connected layer performs weighted sum of the output from stack operation
        fuse_op= self.fusefc1(input1)
        fuse_op=fuse_op.squeeze(-1)
        fuse_op=self.conv1(fuse_op)
        fuse_op=self.bn1(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.conv2(fuse_op)
        fuse_op=self.bn2(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.avgpool1(fuse_op)

        input2=torch.stack([fuse_op,self.hooks['rgb2'],self.hooks['depth2']],dim=-1)
        fuse_op=self.fusefc2(input2)
        fuse_op=fuse_op.squeeze(-1)
        fuse_op=self.conv3(fuse_op)
        fuse_op=self.bn3(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.conv4(fuse_op)
        fuse_op=self.bn4(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.avgpool2(fuse_op)

        input3=torch.stack([fuse_op,self.hooks['rgb3'],self.hooks['depth3']],dim=-1)
        fuse_op=self.fusefc3(input3)
        fuse_op=fuse_op.squeeze(-1)
        fuse_op=self.conv5(fuse_op)
        fuse_op=self.bn5(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.conv6(fuse_op)
        fuse_op=self.bn6(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.avgpool2(fuse_op)

        input4=torch.stack([fuse_op,self.hooks['rgb4'],self.hooks['depth4']],dim=-1)
        fuse_op=self.fusefc4(input4)
        fuse_op=fuse_op.squeeze(-1)
        fuse_op=self.conv7(fuse_op)
        fuse_op=self.bn7(fuse_op)
        fuse_op=self.Relu(fuse_op)
        fuse_op=self.conv8(fuse_op)
        fuse_op=self.bn8(fuse_op)
        fuse_op=self.Relu(fuse_op)

        gap_fusion = self.gavg_pool(fuse_op).squeeze()
        gap_fusion=gap_fusion.view(-1,80)
        fuse_op=self.finalfc(gap_fusion)
        fuse_op = nn.Sigmoid()(fuse_op)

        return  op, op_rgb, op_d,fuse_op


if __name__ == '__main__':
    model = RGBDMH(pretrained=True,num_channels=4)
    input=torch.randn(32, 4,224,224, dtype=torch.float)
    y=model(input)
    for name, param in model.named_parameters():
        print(name)
        print(param.shape)
    import pdb; pdb.set_trace()
    """

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

