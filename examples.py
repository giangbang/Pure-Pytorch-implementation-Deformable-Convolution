import torch
from deform_conv_torch import DeformConv2d

args = {
    "in_channels":4,
    "out_channels":4,
    "kernel_size":3,
    "stride":1,
    "padding":0,
    "dilation": 1,
    "groups" : 2,
    "bias" : True
}

myDeform = DeformConv2d(**args)
device = "cpu"
myDeform.to(device)

# random `input`, `offset` and `mask`
input = torch.rand(1, args["in_channels"], 30, 30,device=device)

# make sure the shape of `offset` and `mask` are correct
offset = torch.rand(1, 36, 28, 28).to(device)
mask = torch.rand((input.shape[0], 18, *offset.shape[2:]),device = device)

myres = myDeform(input, offset, mask)


# pytorch deformanble conv
from  torchvision.ops.deform_conv import DeformConv2d as Deform_torch

deform_code = Deform_torch(**args).to('cuda') # cuda is required

# copy weights and bias
deform_code.weight.data=myDeform.weight.data
deform_code.bias.data=myDeform.bias.data

res = deform_code(input, offset, mask)

# compare result from this implementation and of pytorch
print("difference between this implementation and pytorch's:",(res-myres).square().sum().detach().item())
# difference between this implementation and pytorch's: 6.630807014573747e-13
