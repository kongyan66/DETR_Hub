import sys
import torch
import numpy as np
from pathlib import Path
from util.misc import NestedTensor
from argparse import Namespace

from models.deformable_detr.backbone import build_backbone
from models.deformable_detr.deformable_detr import DeformableDETR
from models.deformable_detr.deformable_transformer import build_deforamble_transformer

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def debug_deformable_detr():
    # 设置CPU设备
    device = torch.device('cpu')
    print(f'使用设备: {device}')

    # 模型配置参数
    config_dict = {
        'backbone': 'resnet50',
        'dilation': False,
        'position_embedding': 'sine',
        'num_classes': 91,
        'num_queries': 100,
        'd_model': 256,
        'enc_layers': 6,
        'hidden_dim': 256,
        'dec_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'nheads': 8,
        'num_feature_levels': 4,
        'dec_n_points': 4,
        'enc_n_points': 4,
        'two_stage': False,
        'with_box_refine': False,
        'lr_backbone': 1e-5,  # 添加缺失的学习率参数
        'masks': False,  # 添加缺失的掩码参数
        'num_feature_levels': 4  # 显式添加特征层级参数
    }
    # 将字典转换为命名空间对象以支持属性访问
    config = Namespace(**config_dict)

    # 构建模型组件
    backbone = build_backbone(config).to(device)
    transformer = build_deforamble_transformer(config).to(device)

    # 创建Deformable DETR模型
    model = DeformableDETR(
        backbone, transformer,
        config.num_classes,
        config.num_queries,
        config.num_feature_levels,
        two_stage=config.two_stage,
      
    ).to(device)

    # 设置模型为评估模式
    model.eval()

    # 创建测试输入 (batch_size=2, 3通道, 640x480)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 640, 480)
    print(f'输入张量形状: {input_tensor.shape}')

    # 创建注意力掩码 (全0表示所有位置可见，1表示填充区域)
    mask = torch.zeros((batch_size, 640, 480), dtype=torch.bool)
    
    # 将输入和掩码封装为NestedTensor
    samples = NestedTensor(input_tensor, mask)

    # 前向传播 - 可以在此处设置断点进行单步调试
    with torch.no_grad():
        # 单步调试时，建议在此处设置断点
        outputs = model(samples)

    # 打印输出信息
    print('\n模型输出内容:')
    print(f'预测框: {outputs["pred_boxes"].shape}')
    print(f'预测分数: {outputs["pred_logits"].shape}')

    return outputs

if __name__ == '__main__':
    set_seed(42)
    debug_deformable_detr()
    print('\n调试脚本执行完成')