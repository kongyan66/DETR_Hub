# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from models.ops.modules.ms_deform_attn import MSDeformAttn



N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])


torch.manual_seed(3)


def test_ms_deform_attn_forward():
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    
    # Create MSDeformAttn model instance (CPU)
    model = MSDeformAttn(d_model=128, n_levels=2, n_heads=4, n_points=2)
    
    # Fixed input parameters
    N, Lq = 1, 2
    
    # Create fixed input tensors
    query = torch.rand(N, Lq, model.d_model)
    reference_points = torch.rand(N, Lq, model.n_levels, 2)
    input_flatten = torch.rand(N, S, model.d_model)
    input_spatial_shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    input_level_start_index = torch.cat((input_spatial_shapes.new_zeros((1, )), input_spatial_shapes.prod(1).cumsum(0)[:-1]))
    
    # Forward pass on CPU
    output = model(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
    
    # Verify output shape
    expected_shape = (N, Lq, model.d_model)
    assert output.shape == expected_shape, f"Unexpected output shape: {output.shape}, expected: {expected_shape}"
    
    # Verify output values with fixed input
    expected_output_sum = 0.3627  # Precomputed expected sum for seed=42
    output_sum = output.sum().item()
    assert torch.allclose(torch.tensor(output_sum), torch.tensor(expected_output_sum), atol=1e-4), \
        f"Output value mismatch: {output_sum:.4f} vs expected {expected_output_sum:.4f}"
    
    print("* MSDeformAttn forward pass test passed with fixed input (CPU)")


if __name__ == '__main__':
    test_ms_deform_attn_forward()



