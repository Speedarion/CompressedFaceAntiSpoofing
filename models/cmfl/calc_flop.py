from network import RGBDMH
import torch
from torch import nn
from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis,flop_count_table,flop_count_str
from torchvision import models
#from pytorch_image_models.timm.models.ghostnet import ghostnet_100,ghostnet_130

model = RGBDMH(4)
#model = ghostnet_100(pretrained=True)
#model = models.densenet161(True)
input = torch.randn(1, 4, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs,params)

flops = FlopCountAnalysis(model, input)
flops.total()
#flops.by_operator()
#flops.by_module()

#flops.by_module_and_operator()
print(flop_count_table(flops))
#print(flop_count_str(flops))

