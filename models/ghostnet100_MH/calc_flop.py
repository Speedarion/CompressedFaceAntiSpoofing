from network import RGBDMH


import torch
from torch import nn
from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis,flop_count_table,flop_count_str

modelA = RGBDMH(pretrained=False)
input = torch.randn(1, 4, 224, 224)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Ghostnet with FC LAYER FUSION has ",count_parameters(modelA),"parameters")


macs, params = profile(modelA, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs,params)

flops = FlopCountAnalysis(modelA, input)
#flops.total()
#flops.by_operator()
#flops.by_module()

#flops.by_module_and_operator()
print(flop_count_table(flops))
#print(flop_count_str(flops))
