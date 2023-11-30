from typing import Dict, List, Tuple, Optional
import numpy as np
from tvm import relax
import tvm
from tvm.ir import IRModule
from tvm.relax.op import linear
from tvm.relax.op.nn import conv2d, conv2d_transpose
from tvm.relax.testing import nn
from tvm.runtime.ndarray import array as tvm_array




class ModuleList(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __iter__(self):
        return iter(self.modules)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __len__(self):
        return len(self.modules)
    
    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x



class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dtype,
        bias=True,
        out_dtype=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features),
            dtype=dtype,
            name="linear_weight"
        )
        if bias:
            self.bis = nn.Parameter(
                (out_features,),
                dtype=dtype if out_dtype is None else out_dtype,
                name="linear_bias"
            )
        else:
            self.bias = None
        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(linear(x, self.weight, self.bias))


# here we ignore conv2d_transpose
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        dtype=None,
        out_dtype=None,
        data_layout='NCHW', 
        kernel_layout='OIHW', 
        out_layout='',
    ):
        self.weight = nn.Parameter((in_channels, out_channels // groups, kernel_size), dtype=dtype, name="conv_weight")

        if bias:
            self.bias = nn.Parameter((out_channels), dtype=dtype, name="conv_bias")
        else:
            self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.dtype = dtype
        self.out_dtype = out_dtype

        self.data_layout = data_layout
        self.kernel_layout = kernel_layout
        self.out_layout = out_layout

    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(conv2d(x, self.weightt, self.stride, self.padding,
            self.dilation, self.groups, self.data_layout, self.kernel_layout,
            self.out_layout, self.out_dtype))




def build_relax_conv(nn_module) -> IRModule:
    bb = relax.BlockBuilder()
    model = nn_module(3, 64, 3)
    with bb.function("main"):
        input = nn.Placeholder((32,32,3), dtype="float32", name="input")
        with bb.dataflow():
            logits = model(input)
            params = [input] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)
    return bb.get()


def build_relax_Linear(nn_module) -> IRModule:
    bb = relax.BlockBuilder()
    model = nn_module(784, 128, "float32")
    with bb.function("main"):
        input = nn.Placeholder((1, 784), dtype="float32", name="input")
        with bb.dataflow():
            logits = model(input)
            params = [input] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)
    return bb.get()


def optimize_and_deploy(mod):
    mod = relax.pipeline.get_pipeline()(mod)
    ex = relax.build(mod, "llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    import torch
    import torch.nn as nn
    import numpy as np
    input_np = np.random.rand(32,32,3).astype("float32")
    weight_np = np.random.rand(64, 3, 3, 3).astype("float32")
    bias_np   = np.random.rand(64).astype("float32")
    
    tvm_nd_arrays = [tvm.nd.array(np_array, device=tvm.cpu()) for np_array in [input_np, weight_np, bias_np]]
    # call into the runnable function converted from IRModule
    nd_res = vm["main"](*tvm_nd_arrays)

    print("IRModule execution result:", nd_res.numpy())
    conv = nn.Conv2d(3, 64, 3)
    conv.weight.data = torch.from_numpy(weight_np)
    conv.bias.data   = torch.from_numpy(bias_np)
    t_result = conv(torch.from_numpy(input_np))
    print("numpy execution result:", (t_result))
