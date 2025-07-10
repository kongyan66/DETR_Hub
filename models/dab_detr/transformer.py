# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

class MLP(nn.Module):
    """多层感知机(MLP)，也称为前馈网络(FFN)

    输入: 任意形状张量
    输出: 经过多层线性变换和激活函数的张量，形状与输入一致
"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  # 记录网络层数
        # 创建隐藏层维度列表，除最后一层外均为hidden_dim
        h = [hidden_dim] * (num_layers - 1)
        # 创建线性层列表，实现从输入到输出的维度变换
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 遍历每一层线性层
        for i, layer in enumerate(self.layers):
            # 对除最后一层外的所有层应用ReLU激活函数
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x  # 返回最终输出，形状与输入x一致

def gen_sineembed_for_position(pos_tensor, d_model=256):
    """生成位置的正弦嵌入

    参数:
        pos_tensor: 位置张量，形状为 (n_query, bs, 2) 或 (n_query, bs, 4)
                    2D坐标格式为 [x, y]，4D格式为 [x, y, w, h]
        d_model: 嵌入维度，默认256

    返回:
        pos: 正弦位置嵌入，形状为 (n_query, bs, d_model)
"""
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi  # 角度缩放因子，将坐标值转换到[0, 2π]范围
    # 创建频率维度张量，用于生成不同频率的正弦/余弦函数
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    # 计算频率缩放因子，遵循Transformer中的位置编码公式
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    
    # 提取x坐标并进行缩放
    x_embed = pos_tensor[:, :, 0] * scale  # 形状: (n_query, bs)
    # 提取y坐标并进行缩放
    y_embed = pos_tensor[:, :, 1] * scale  # 形状: (n_query, bs)
    
    # 计算x坐标的位置编码，增加维度以便后续处理
    pos_x = x_embed[:, :, None] / dim_t  # 形状: (n_query, bs, d_model//2)
    # 计算y坐标的位置编码，增加维度以便后续处理
    pos_y = y_embed[:, :, None] / dim_t  # 形状: (n_query, bs, d_model//2)
    
    # 对x坐标嵌入交替应用sin和cos函数，并展平维度
    # 奇数索引使用cos，偶数索引使用sin
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # 形状: (n_query, bs, d_model//2)
    # 对y坐标嵌入交替应用sin和cos函数，并展平维度
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)  # 形状: (n_query, bs, d_model//2)
    
    # 根据输入位置张量的维度选择不同的处理方式
    if pos_tensor.size(-1) == 2:
        # 对于2D坐标，拼接y和x的位置嵌入
        pos = torch.cat((pos_y, pos_x), dim=2)  # 形状: (n_query, bs, d_model)
    elif pos_tensor.size(-1) == 4:
        # 提取宽度w并进行缩放
        w_embed = pos_tensor[:, :, 2] * scale  # 形状: (n_query, bs)
        # 计算宽度的位置编码
        pos_w = w_embed[:, :, None] / dim_t  # 形状: (n_query, bs, d_model//2)
        # 对宽度嵌入交替应用sin和cos函数
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)  # 形状: (n_query, bs, d_model//4)

        # 提取高度h并进行缩放
        h_embed = pos_tensor[:, :, 3] * scale  # 形状: (n_query, bs)
        # 计算高度的位置编码
        pos_h = h_embed[:, :, None] / dim_t  # 形状: (n_query, bs, d_model//2)
        # 对高度嵌入交替应用sin和cos函数
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)  # 形状: (n_query, bs, d_model//4)

        # 拼接y, x, w, h的位置嵌入
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)  # 形状: (n_query, bs, d_model)
    else:
        # 不支持的位置张量维度
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos  # 返回最终的位置嵌入张量


class Transformer(nn.Module):
    """DAB-DETR的Transformer模型，包含编码器和解码器

    主要功能: 将图像特征和位置嵌入转换为目标检测所需的特征表示
    数据流程: 图像特征 -> 编码器 -> 解码器 -> 目标特征和边界框预测
"""

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):
        """Transformer模型初始化

        参数:
            d_model: 模型维度，默认512
            nhead: 注意力头数，默认8
            num_queries: 查询数量，默认300
            num_encoder_layers: 编码器层数，默认6
            num_decoder_layers: 解码器层数，默认6
            dim_feedforward: 前馈网络维度，默认2048
            dropout: dropout比率，默认0.1
            activation: 激活函数类型，默认"relu"
            normalize_before: 是否在注意力前进行归一化，默认False
            return_intermediate_dec: 是否返回解码器中间层输出，默认False
            query_dim: 查询维度，默认4
            keep_query_pos: 是否保留查询位置，默认False
            query_scale_type: 查询缩放类型，默认'cond_elewise'
            num_patterns: 模式数量，默认0
            modulate_hw_attn: 是否调制高宽注意力，默认True
            bbox_embed_diff_each_layer: 是否每层使用不同的边界框嵌入，默认False
        """
        super().__init__()

        # 创建编码器层
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 创建编码器归一化层
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # 初始化编码器
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 创建解码器层
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        # 创建解码器归一化层
        decoder_norm = nn.LayerNorm(d_model)
        # 初始化解码器
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        # 重置参数
        self._reset_parameters()
        # 验证查询缩放类型
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        # 保存模型参数
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        # 验证模式数量类型
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        # 初始化模式嵌入（如果需要）
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)  # 形状: (num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        """Transformer前向传播

        参数:
            src: 图像特征，形状为 (bs, c, h, w)
            mask: 掩码张量，形状为 (bs, h, w)
            refpoint_embed: 参考点嵌入，形状为 (num_queries, bs, query_dim)
            pos_embed: 位置嵌入，形状为 (bs, c, h, w)

        返回:
            hs: 解码器输出特征，形状为 (num_decoder_layers, bs, num_queries, d_model)
            references: 更新后的参考点，形状为 (num_decoder_layers, bs, num_queries, query_dim)
        """
        # 展平NxCxHxW为HWxNxC
        bs, c, h, w = src.shape  # 获取输入特征形状
        # 将特征从 (bs, c, h, w) 展平为 (bs, c, h*w) 并转置为 (h*w, bs, c)
        src = src.flatten(2).permute(2, 0, 1)  # 形状: (L, bs, c)，其中L = h*w
        # 处理位置嵌入，形状从 (bs, c, h, w) 变为 (L, bs, c)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # 形状: (L, bs, c)
        # 扩展参考点嵌入维度并复制，形状从 (num_queries, query_dim) 变为 (num_queries, bs, query_dim)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1) # 形状: (num_queries, bs, query_dim)
        # 将掩码展平，形状从 (bs, h, w) 变为 (bs, h*w)
        mask = mask.flatten(1) # 形状: (bs, L)
        # 通过编码器获取记忆特征
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # 形状: (L, bs, c)

        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]  # 获取查询数量
        # 初始化目标查询
        if self.num_patterns == 0:
            # 如果没有模式，初始化为零张量，形状: (num_queries, bs, d_model)
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            # 如果有模式，使用模式嵌入初始化目标查询
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # 形状: (n_q*n_pat, bs, d_model)
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # 形状: (n_q*n_pat, bs, d_model)
            # import ipdb; ipdb.set_trace()
        # 通过解码器获取输出特征和参考点
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references  # 返回解码器输出和参考点



class TransformerEncoder(nn.Module):
    """Transformer编码器

    功能: 对输入图像特征进行自注意力处理，增强特征表示
    结构: 多层TransformerEncoderLayer堆叠而成
"""

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        """编码器初始化

        参数:
            encoder_layer: 编码器层实例
            num_layers: 编码器层数
            norm: 归一化层，默认None
            d_model: 模型维度，默认256
        """
        super().__init__()
        # 创建编码器层的深度拷贝列表
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 创建查询缩放MLP，用于调整位置嵌入的缩放因子
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """编码器前向传播

        参数:
            src: 输入特征，形状为 (L, bs, d_model)
            mask: 注意力掩码，默认None
            src_key_padding_mask: 输入填充掩码，默认None
            pos: 位置嵌入，形状为 (L, bs, d_model)

        返回:
            output: 编码后的特征，形状为 (L, bs, d_model)
        """
        output = src  # 初始化输出为输入特征

        # 遍历每一层编码器
        for layer_id, layer in enumerate(self.layers):
            # 计算位置缩放因子，用于调整内容和位置相似度
            pos_scales = self.query_scale(output)  # 形状: (L, bs, d_model)
            # 通过编码器层处理，应用位置缩放
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        # 应用最终归一化（如果指定）
        if self.norm is not None:
            output = self.norm(output)

        return output  # 返回编码后的特征


class TransformerDecoder(nn.Module):
    """Transformer解码器

    功能: 将编码器输出和查询嵌入转换为目标检测特征
    特点: 包含参考点更新机制和位置嵌入变换
"""

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        """解码器初始化

        参数:
            decoder_layer: 解码器层实例
            num_layers: 解码器层数
            norm: 归一化层，默认None
            return_intermediate: 是否返回中间层输出，默认False
            d_model: 模型维度，默认256
            query_dim: 查询维度，默认2
            keep_query_pos: 是否保留查询位置，默认False
            query_scale_type: 查询缩放类型，默认'cond_elewise'
            modulate_hw_attn: 是否调制高宽注意力，默认False
            bbox_embed_diff_each_layer: 是否每层使用不同边界框嵌入，默认False
        """
        super().__init__()
        # 创建解码器层的深度拷贝列表
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "必须返回中间层输出"
        self.query_dim = query_dim

        # 验证查询缩放类型
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        # 根据缩放类型初始化查询缩放网络
        if query_scale_type == 'cond_elewise':
            # 代表每个元素都根据条件来生成，意思是每个 scale 向量都根据 Decoder 每层的 output 来生成(维度上一一对应)
            self.query_scale = MLP(d_model, d_model, d_model, 2)  # 元素级条件缩放
        elif query_scale_type == 'cond_scalar':
            # 代表根据条件来生成标量，即每个 scale 向量会根据 Decoder 每层的 output 来生成，但它是个标量
            self.query_scale = MLP(d_model, d_model, 1, 2)  # 标量条件缩放
        elif query_scale_type == 'fix_elewise':
            # 不根据 Decoder 的 output 来生成，而是用它来在 Embedding 中做 lookup
            self.query_scale = nn.Embedding(num_layers, d_model)  # 固定元素级缩放
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        # 创建参考点头，用于处理位置嵌入
        # ref_point_head 的输入是参考点的位置嵌入，因此输入维度就是位置嵌入的维度，输出用作位置 query
        # 由于位置先验是 4d(query_dim=4)的，于是由它经历正余弦编码后得到的 embedding 维度是 2 * d_model，
        # 因此这里用这个 ref_point_head 将其映射回 d_model 维度
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None  # 边界框嵌入网络，后续初始化
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn  # 是否调制高宽注意力
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer  # 是否每层使用不同边界框嵌入

        # 如果需要调制高宽注意力，创建参考锚点头
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)  # 输出宽度和高度调制因子

        # 如果不保留查询位置，除第一层外禁用交叉注意力的查询位置投影
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, query_dim   pos_query
                ):
        """解码器前向传播

        参数:
            tgt: 查询嵌入张量，形状为 (num_queries, bs, d_model)
            memory: 编码器输出特征，形状为 (L, bs, d_model)，L为特征点数
            tgt_mask: 查询掩码张量，默认None
            memory_mask: 记忆特征掩码，默认None
            tgt_key_padding_mask: 查询填充掩码，默认None
            memory_key_padding_mask: 记忆特征填充掩码，默认None
            pos: 位置嵌入张量，形状为 (L, bs, d_model)
            refpoints_unsigmoid: 未经过sigmoid的参考点坐标，形状为 (num_queries, bs, query_dim)

        返回:
            若return_intermediate=True: (中间输出列表, 中间参考点列表)
            否则: 最终输出张量，形状为 (1, bs, num_queries, d_model)
        """
        # (num_queries,bs,d_model)
        output = tgt

        # 收集每一层的输出结果
        intermediate = []
        # 将 x,y,w,h 缩放到 0~1
        reference_points = refpoints_unsigmoid.sigmoid()
        # 收集每层的参考点，除第一层外，每层均会进行校正
        ref_points = [reference_points] 

        for layer_id, layer in enumerate(self.layers):
            # i. 获得位置 query: 参考点->正余弦编码获得参考点的位置嵌入->2层 MLP 获得位置 query
            # (num_queries, batch_size, 4)
            obj_center = reference_points[..., :self.query_dim]
            # Get sine embedding for the query vector
            # 在4个维度(x,y,w,h)独立进行位置编码然后拼接在一起
            # (num_queries,bs,4*128=4*(d_model//2))
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # 位置 query (num_queries,bs,d_model)
            query_pos = self.ref_point_head(query_sine_embed) 

            # ii. 基于当前层的 output 生成 transformation，对参考点的位置嵌入做变换，将内容与位置信息结合
            if self.query_scale_type != 'fix_elewise':
                # For the first decoder layer, we do not apply transformation over p_s
                # 因为在第一层中的 output 是初始化的 query(以上传参过来的 tgt)，
                # 所以基于它来生成对位置嵌入做变换的向量(pos_transformation)没有意义
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # Apply transformation
            # 注意这里做了截断，在最后一维截取前 d_model 个维数
            # (num_queries,bs,d_model)
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation  

            # modulated HW attentions
            if self.modulate_hw_attn:
                # 基于当前层的 output 生成 x, y 坐标的调制参数(向量)，对应于 paper 公式中的 w_{q,ref} & h_{q,ref}
                # (num_queries,bs,2)
                refHW_cond = self.ref_anchor_head(output).sigmoid()
                # 分别调制 x, y 坐标并处以 w, h 归一化，从而将尺度信息注入到交叉注意力中
                # 后 self.d_model // 2 个维度对应 x 坐标，前 self.d_model // 2 个维度对应 y 坐标；
                # query_sine_embed[..., self.d_model // 2:] 对应 paper 公式的 PE(x)
                # query_sine_embed[..., :self.d_model // 2] 对应 paper 公式的 PE(y)
                # obj_center[..., 2] 是宽，对应 paper 公式的 w_q，obj_center[..., 3] 是高，对应 paper 公式的 h_q
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            # 解码
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # Iter update
            # 更新参考点
            if self.bbox_embed is not None:
                # 生成 bbox 的偏移量
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)

                # 由于参考点坐标是经过了 sigmoid，因此这里先反 sigmoid 再加上偏移量
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # 更新参考点后重新经过 sigmoid 缩放
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                # 最后一层的参考点会在外层模型中由整个 transformer 的输出经过 bbox_embed 得到偏移量，然后更新
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                
                # 作者说(本人说的哦) detach() 是因为让梯度的流通更友好，它想让每层的梯度仅受该层的输出影响
                reference_points = new_reference_points.detach()

                # 那么参考点(嵌入向量)如何被训练而学习？
                # 在 Decoder 第一层，参考点进来时，由其生成了 query 位置嵌入向量(query_sine_embed) & 位置 query(query_pos)，
                # 改层的 output 与它们都相关联。同时，由于每层的 input 都是前一层的 output，
                # 因此能够通过 loss 计算，最后反向传播得到梯度。

                # 另外，对于校正模块(box_embed)的优化，是通过隐层向量输入到它里面生成偏移量、结合最后一层解码出来的参考点得到
                # 对象的位置，最后与标签计算 loss 使得梯度得以传导过来。

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (num_layers,bs,num_queries,4)
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (1,bs,num_queries,4) 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        # 这里增加一个维度是为了适配返回中间层(return_intermediate=True)的情况，
        # 使得当 return_intermediate=False 时，len(output)=0，代表仅返回最后一层的结果。
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层

    功能: 实现单个编码器层的自注意力和前馈网络处理
    结构: 自注意力子层 -> 前馈网络子层，均带残差连接和层归一化
"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """编码器层初始化

        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力前进行归一化
        """
        super().__init__()
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 输入投影
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 输出投影

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈网络后归一化
        # dropout层
        self.dropout1 = nn.Dropout(dropout)  # 自注意力输出dropout
        self.dropout2 = nn.Dropout(dropout)  # 前馈网络输出dropout

        self.activation = _get_activation_fn(activation)  # 获取激活函数
        self.normalize_before = normalize_before  # 归一化位置标志

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """为张量添加位置嵌入

        参数:
            tensor: 输入张量
            pos: 位置嵌入张量
        返回:
            添加位置嵌入后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """编码器层前向传播

        参数:
            src: 输入特征，形状为 (L, bs, d_model)
            src_mask: 注意力掩码
            src_key_padding_mask: 输入填充掩码
            pos: 位置嵌入，形状为 (L, bs, d_model)

        返回:
            编码后的特征，形状为 (L, bs, d_model)
        """
        # 生成带位置嵌入的查询和键
        q = k = self.with_pos_embed(src, pos)  # 形状: (L, bs, d_model)
        # 自注意力计算
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  # 形状: (L, bs, d_model)
        # 残差连接和dropout
        src = src + self.dropout1(src2)
        # 归一化
        src = self.norm1(src)
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 形状: (L, bs, d_model)
        # 残差连接和dropout
        src = src + self.dropout2(src2)
        # 归一化
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层

    功能: 实现单个解码器层的自注意力、交叉注意力和前馈网络处理
    结构: 自注意力子层 -> 交叉注意力子层 -> 前馈网络子层，均带残差连接和层归一化
"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        """解码器层初始化

        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力前进行归一化
            keep_query_pos: 是否保留查询位置嵌入
            rm_self_attn_decoder: 是否移除解码器自注意力
        """
        super().__init__()
        # 解码器自注意力（可选）
        if not rm_self_attn_decoder:
            # 自注意力投影层
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)  # 查询内容投影
            self.sa_qpos_proj = nn.Linear(d_model, d_model)      # 查询位置投影
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)  # 键内容投影
            self.sa_kpos_proj = nn.Linear(d_model, d_model)      # 键位置投影
            self.sa_v_proj = nn.Linear(d_model, d_model)         # 值投影
            # 自注意力模块（值维度为d_model）
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)  # 自注意力后归一化
            self.dropout1 = nn.Dropout(dropout)  # 自注意力输出dropout

        # 解码器交叉注意力
        # 交叉注意力投影层
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)  # 查询内容投影
        self.ca_qpos_proj = nn.Linear(d_model, d_model)      # 查询位置投影
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)  # 键内容投影
        self.ca_kpos_proj = nn.Linear(d_model, d_model)      # 键位置投影
        self.ca_v_proj = nn.Linear(d_model, d_model)         # 值投影
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)  # 正弦位置嵌入投影
        # 交叉注意力模块（输入维度为2*d_model，值维度为d_model）
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead  # 注意力头数
        self.rm_self_attn_decoder = rm_self_attn_decoder  # 是否移除自注意力标志

        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 输入投影
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 输出投影

        # 归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络后归一化
        # dropout层
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力输出dropout
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络输出dropout

        self.activation = _get_activation_fn(activation)  # 获取激活函数
        self.normalize_before = normalize_before  # 归一化位置标志
        self.keep_query_pos = keep_query_pos  # 保留查询位置标志

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """为张量添加位置嵌入

        参数:
            tensor: 输入张量
            pos: 位置嵌入张量
        返回:
            添加位置嵌入后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        """解码器层前向传播

        参数:
            tgt: 查询嵌入，形状为 (num_queries, bs, d_model)
            memory: 编码器输出特征，形状为 (L, bs, d_model)
            tgt_mask: 查询掩码
            memory_mask: 记忆特征掩码
            tgt_key_padding_mask: 查询填充掩码
            memory_key_padding_mask: 记忆特征填充掩码
            pos: 位置嵌入，形状为 (L, bs, d_model)
            query_pos: 查询位置嵌入，形状为 (num_queries, bs, d_model)
            query_sine_embed: 查询正弦位置嵌入，形状为 (num_queries, bs, d_model)
            is_first: 是否为第一层解码器

        返回:
            更新后的查询嵌入，形状为 (num_queries, bs, d_model)
        """               
        # ========== Begin of Self-Attention =============
        # attention->dropout->residual->norm
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer, zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            # num_queries, bs, n_model = q_content.shape
            # hw, _, _ = k_content.shape

            # 内容+位置
            q = q_content + q_pos
            k = k_content + k_pos

            # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            # 残差连接
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # attention->dropout->residual->norm
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        # 第一层由于没有足够的位置信息，因此默认要加上位置部分
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        # 将 query & key 的内容与通过正余弦位置编码得到的位置部分拼接，
        # 从而两者在交叉注意力中做交互是能够实现 内容与位置分别独立做交互，即：
        # q_content <-> k_content; q_position <-> k_position
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        # query_sine_embed 由 4d anchor box 经历正余弦位置编码而来，实现了与 key 一致的位置编码方式
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # 这里有个细节要注意，就是在拼接(concat)前要先将最后一维按注意力头进行划分，这样才能将各个头部的维度对应拼接
        # 否则，就会导致前面一些头部全部都是 q 的部分，而后面一些头部则全是 query_sine_embed 的部分。
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)  # 非常关键的步骤，将query的conent和position解耦开  

        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
        tgt2 = self.cross_attn(
            query=q,
            key=k,
            value=v, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        # linear->activation->dropout->linear
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # dropout->residual->norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt



def _get_clones(module, N):
    """创建模块的N个深度拷贝

    功能: 为编码器/解码器创建多个相同配置的层
    参数:
        module: 要拷贝的基础模块
        N: 拷贝数量
    返回:
        nn.ModuleList: 包含N个模块拷贝的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
