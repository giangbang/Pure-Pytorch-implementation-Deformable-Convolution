import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def _sampling(x: torch.Tensor, coord: torch.Tensor):
    b, c, h, w = x.shape
    # coord : b, g, h, w, k, 2
    _, g, oh, ow, k, _ = coord.shape

    coord = coord.reshape(b, g, 1, -1, 2).repeat(1,1,c//g,1,1).view(b*c, -1, 2)
    outside = torch.bitwise_or((coord[..., 0] >= h) + (coord[..., 0] < 0),
              (coord[..., 1] >= w) + (coord[..., 1] < 0))
    
    device = coord.device
    
    indx = coord[..., 0]*w + coord[..., 1] + \
            (h*w*torch.arange(c*b, device=device)).view(-1, 1)
    
    x = x.view(-1).index_select(0, indx.view(-1).clamp(0, b*c*h*w-1))
    x[outside.view(-1)] = 0
    # [b, c, h, w, k]
    return x.reshape(b, -1, oh, ow, k)

def _interpolate(x: torch.Tensor, y: torch.Tensor, frac: torch.Tensor):
    '''
    Linear interpolation between x and y
    Sub-routine for bi-linear interpolation
    x, y: [b, c, h, w, k]
    frac: [b, g, h, w, k]
    '''
    assert x.shape == y.shape
    b, c, h, w, k = x.shape
    g = frac.shape[1]
    x = x.view(b, g, c//g, h, w, k)
    y = y.view(b, g, c//g, h, w, k)
    frac = frac.view(b, g, 1, h, w, k)
    res =  x + frac*(y-x)
    return res.view(b, c, h, w, k)

def _output_size(input, weight, padding, dilation, stride):
    channels = weight.size(0)
    output_size = (input.size(0), channels)
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = padding[d]
        kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
        stride_ = stride[d]
        output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError(
            "convolution input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
    return output_size

def deform_conv2d(
    input: Tensor, 
    offset: Tensor, 
    weight: Tensor, 
    bias: Optional[Tensor] = None, 
    stride: Tuple[int, int] = (1, 1), 
    padding: Tuple[int, int] = (0, 0), 
    dilation: Tuple[int, int] = (1, 1), 
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Pytorch equivalent implementation of `torchvision.ops.deform_conv2d`,
    The outputs of this function and that of torchvision's are the exactly same
    """
    
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    use_mask = mask is not None
    use_bias = bias is not None

    out_channels, _, kernel_height, kernel_width = weight.shape
    batch_size, in_channels, in_height, in_width = input.shape
    groups = offset.shape[1] // 2 // kernel_height // kernel_width
    out_height, out_width = offset.shape[-2:]
    k = int(kernel_height * kernel_width)
    kernel_size = weight.shape[-2:]
    device = input.device

    output_size_shouldbe = _output_size(input, weight, padding, dilation, stride)
    assert output_size_shouldbe == (batch_size, out_channels, out_height, out_width), (
        "The shape of `offset` is incorrect? "
        "We expect (batch_size, out_channels, out_height, out_width) to be "
        f"{output_size_shouldbe}"
        ", but we get: "
        f"({batch_size} {out_channels} {out_height} {out_width})."
    )

    # indices of padded input
    grid_i, grid_j = torch.meshgrid(
        torch.arange(-pad_h, in_height + pad_h, device=device),
        torch.arange(-pad_w, in_width + pad_w, device=device), 
        indexing='ij'
    )

    grid_coord = torch.cat((grid_i.unsqueeze(2), grid_j.unsqueeze(2)), 2).float() # w,h,2

    # im2col stride trick
    grid_coord_im2col = torch.as_strided(grid_coord, 
                size = (out_height, out_width, *kernel_size, 2),
                stride=(grid_coord.stride(0) * stride_h, grid_coord.stride(1) * stride_w, 
                        grid_coord.stride(0) * dil_h, grid_coord.stride(1) * dil_w, 
                        grid_coord.stride(2)) 
    )

    grid_coord_im2col = grid_coord_im2col.reshape(1, 1, out_height, out_width, -1, 2)

    offset = offset.view(batch_size, groups, k, 2, out_height, out_width)
    offset = offset.permute([0, 1, 4, 5, 2, 3])
    coord = (offset + grid_coord_im2col)

    coord_lt = coord.floor().long() # b, g, h, w, k, 2
    coord_rb = coord_lt + 1
    coord_rt = torch.stack([coord_lt[..., 0], coord_rb[..., 1]], -1)
    coord_lb = torch.stack([coord_rb[..., 0], coord_lt[..., 1]], -1)

    # [b, c, h, w, k]
    vals_lt = _sampling(input, coord_lt.detach())
    vals_rb = _sampling(input, coord_rb.detach())
    vals_lb = _sampling(input, coord_lb.detach())
    vals_rt = _sampling(input, coord_rt.detach())
    
    # bi-linear interpolation
    fract = coord - coord_lt.float() # [b, g, h, w, k, 2]
    vals_t = _interpolate(vals_lt, vals_rt, fract[..., 1])
    vals_b = _interpolate(vals_lb, vals_rb, fract[..., 1])
    
    mapped_vals = _interpolate(vals_t, vals_b, fract[..., 0])
    mapped_vals = mapped_vals.permute([1, 4, 0, 2, 3]) # [c, k, b, h, w]

    if use_mask: 
        # mask: [g, 1, k, b, h, w]
        mask = mask.permute([1,0,2,3]).view(groups, 1, k, batch_size, out_height, out_width)
        mapped_vals = mapped_vals.view(groups, in_channels//groups, k, batch_size, out_height, out_width)
        assert len(mapped_vals.shape) == len(mask.shape), f"{mapped_vals.shape} {mask.shape}"
        mapped_vals = mapped_vals * mask
        mapped_vals = mapped_vals.view(in_channels, k, batch_size, out_height, out_width)
    mapped_vals = mapped_vals.reshape(groups, -1, batch_size * out_height * out_width) # g x C//g k x BHW

    output = torch.matmul(weight.view(groups, out_channels//groups, -1), mapped_vals) 
    output = output.view(out_channels, batch_size, out_height, out_width)  # C' x BHW

    if use_bias:
        output = output + bias.view(-1, 1, 1, 1)
    
    return output.permute([1,0,2,3])

class DeformConv2d(nn.Module):
    """
    This class is copied from pytorch, see 
    https://pytorch.org/vision/main/_modules/torchvision/ops/deform_conv.html#DeformConv2d
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s
