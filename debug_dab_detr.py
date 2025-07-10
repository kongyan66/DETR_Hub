# 导入必要的库和模块
import sys
import torch
import numpy as np
from pathlib import Path
from argparse import Namespace

# 导入项目内部模块
from util.misc import NestedTensor  # 用于封装输入张量和掩码
from models.dab_detr.backbone import build_backbone  # 骨干网络构建函数
from models.dab_detr.DABDETR import DABDETR  # DAB-DETR模型类
from models.dab_detr.transformer import build_transformer  # Transformer构建函数

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def debug_dab_detr():
    # 设置CPU设备
    device = torch.device('cpu')
    print(f'使用设备: {device}')

    # 模型核心配置参数
    config_dict = {
        'backbone': 'resnet50',               # 骨干网络类型
        'dilation': False,                    # 是否使用空洞卷积
        'position_embedding': 'sine',         # 位置编码类型
        'num_classes': 91,                    # COCO数据集类别数(含背景)
        'num_queries': 100,                   # 查询向量数量
        'd_model': 256,                       # Transformer隐藏层维度
        'enc_layers': 6,                      # 编码器层数
        'hidden_dim': 256,                    # 隐藏层维度
        'dec_layers': 6,                      # 解码器层数
        'dim_feedforward': 1024,              # 前馈网络维度
        'pe_temperatureH': 10000,             # 高度方向位置编码温度参数
        'pe_temperatureW': 10000,             # 宽度方向位置编码温度参数
        'transformer_activation': 'relu',     # Transformer激活函数
        'num_patterns': 4,                    # 模式数量
        'num_feature_levels': 4,              # 特征层级数量
        'pre_norm': False,                    # 是否使用PreNorm
        'dropout': 0.1,                       # Dropout比率
        'nheads': 8,                          # 注意力头数
        'lr_backbone': 1e-5,                  # 骨干网络学习率
        'masks': False,                       # 是否使用掩码
        'query_dim': 4,                       # 查询维度(4表示边界框)
        'bbox_embed_diff_each_layer': False,  # 是否每层使用不同的边界框嵌入
        'iter_update': True                   # 是否迭代更新边界框
    }
    # 将字典转换为命名空间对象以支持属性访问
    config = Namespace(**config_dict)

    # 构建模型组件
    backbone = build_backbone(config).to(device)
    transformer = build_transformer(config).to(device)

    # 创建DAB-DETR模型
    model = DABDETR(
        backbone, transformer,
        config.num_classes,                  # 类别数量
        config.num_queries,                  # 查询数量
        config.num_feature_levels,           # 特征层级数量
        config.dec_layers,                   # 解码器层数
        # 注意: 根据DABDETR构造函数，此处省略了默认参数aux_loss=False
    ).to(device)

    # 设置模型为评估模式(关闭 dropout 和 batch normalization)
    model.eval()

    # 创建测试输入 (batch_size=2, 3通道RGB图像, 分辨率640x480)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 640, 480)
    print(f'输入张量形状: {input_tensor.shape}')

    # 创建注意力掩码 (全0表示所有位置可见，1表示填充区域)
    # 由于输入是随机生成的，这里没有实际填充区域，故全为False
    mask = torch.zeros((batch_size, 640, 480), dtype=torch.bool)
    
    # 将输入和掩码封装为NestedTensor，这是模型期望的输入格式
    samples = NestedTensor(input_tensor, mask)

    # 前向传播 - 可以在此处设置断点进行单步调试
    # 使用torch.no_grad()禁用梯度计算，节省内存并加速计算
    with torch.no_grad():
        # 单步调试时，建议在此处设置断点
        outputs = model(samples)

    # 打印输出信息，验证模型输出格式是否正确
    print('\n模型输出内容:')
    print(f'预测框: {outputs["pred_boxes"].shape}')  # 形状: [batch_size, num_queries, 4]
    print(f'预测分数: {outputs["pred_logits"].shape}')  # 形状: [batch_size, num_queries, num_classes]

    return outputs

if __name__ == '__main__':
    set_seed(42)
    debug_dab_detr()
    print('\n调试脚本执行完成')