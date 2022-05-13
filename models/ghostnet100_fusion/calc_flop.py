from network import RGBDMH as fusionA
from networkB import RGBDMH as fusionB
from networkC import RGBDMH as fusionC

import torch
from torch import nn
from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis,flop_count_table,flop_count_str

modelA = fusionA(4)
modelB = fusionB(4)
modelC = fusionC(4)

input = torch.randn(1, 4, 224, 224)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Ghostnet with fusion A has ",count_parameters(modelA),"parameters")
print("Ghostnet with fusion B has ",count_parameters(modelB),"parameters")
print("Ghostnet with fusion C has ",count_parameters(modelC),"parameters")

macs, params = profile(modelC, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs,params)

flops = FlopCountAnalysis(modelC, input)
#flops.total()
#flops.by_operator()
#flops.by_module()

#flops.by_module_and_operator()
print(flop_count_table(flops))
#print(flop_count_str(flops))
