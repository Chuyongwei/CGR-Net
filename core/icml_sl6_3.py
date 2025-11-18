import torch
import torch.nn as nn
from loss import batch_episym
import numpy as np
import torch.nn.functional as F

'''
@Title icml_sl6_3
@Date: 2025/11/13
@Author: ChuyongWei
@Version: 1.1
@Description:
基于icml_m4_f6

首先首部的丰富一下空间特征，然后使用向量作为权重去去做一个自适应的融合
dmfc改为双路的自适应的卷积方式
6：4284108
8：4520268
首尾加入MBFFN

@Evaluation
map5 73.95
'''

class SE(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):  # num_channels=64
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = nn.Conv2d(num_channels, num_channels,
                            kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.in0(input_tensor)
        x = self.bn0(x)
        x = self.relu(x)  # b,128,2000,1
        input_tensor = self.conv0(x)
        squeeze_tensor = input_tensor.view(
            batch_size, num_channels, -1).mean(dim=2)  # 对每个通道求平均值  b,128
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))  # b,64
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))  # b,128
        a, b = squeeze_tensor.size()  # a:batch_size, b:128
        # b,128,2000,1    b,128,1,1---->b,128,2000,1
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class MBSE(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):  # planes=64
        super(MBSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes * 2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes * 2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, SE_channel, kernel_size=1, bias=True)
        self.in1 = nn.InstanceNorm2d(inplanes, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = SE(SE_channel)
        self.conv3 = nn.Conv2d(SE_channel, planes * 2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(SE_channel, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(SE_channel)
        self.conv_branch1_1 = nn.Sequential(
            nn.InstanceNorm2d(inplanes, eps=1e-3),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, planes * 2, kernel_size=1),
        )
        self.conv_merge = nn.Sequential(
            nn.InstanceNorm2d(planes * 2, eps=1e-3),
            nn.BatchNorm2d(planes * 2),
            nn.GELU(),
            nn.Conv2d(planes * 2, planes * 2, kernel_size=1),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.in1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.in3(out)
        out = self.bn3(out)
        out = self.conv3(out)
        branch_out = self.conv_branch1_1(x)
        # branch_out = self.conv_branch1_2(branch_out)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out = out + branch_out + residual
        out = self.conv_merge(out)
        return out

# Cross-Stage Multi-Graph Consensus Module, CSMGC
class CSMGC(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.annular_convolution = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2), stride=(1, 2)),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2)),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
            )
        self.MLP = MLPs(in_channels *2 , out_channels)


    def forward(self, Stage_1, Stage_2):
        S1_graph = get_graph_feature(Stage_1, k=2)
        S2_graph = get_graph_feature(Stage_2, k=2)
        Combine_Stage_Graph = torch.cat([S1_graph, S2_graph], dim=-1)
        ANN_out = self.MLP(self.annular_convolution(Combine_Stage_Graph))
        out = ANN_out + Stage_2
        return out

class MLPs(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MBFFN_v2(nn.Module):
    """
    一个优化后的 MBFFN 模块，具有以下特性:
    1. 为全局平均/最大池化共享一个 MLP，以减少参数。
    2. 添加了灵活的 shortcut 连接，允许 in_channels != out_channels。
    3. (可选) 将 nn.Sigmoid() 替换为 nn.Hardsigmoid() 以提高计算效率。
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super(MBFFN_v2, self).__init__()

        # 确保中间通道数至少为1 (避免 reduction 过大导致为0)
        inter_channels = max(int(in_channels // reduction), 1)

        # ----------------- 1. 灵活的 Shortcut -----------------
        # 如果通道数不匹配，使用 1x1 卷积；否则，使用恒等映射
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # ----------------- 2. 核心卷积 -----------------
        # 输入和输出卷积（用于特征转换）
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # ----------------- 3. 局部（空间）注意力流 -----------------
        # 这是一个在每个空间位置上独立应用的MLP
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # ----------------- 4. 全局（通道）注意力流 -----------------
        # 两个池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP (替换了原版中冗余的 global_att 和 global_att_max)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # ----------------- 5. 激活门控 -----------------
        # 使用 Hardsigmoid 更快，但 Sigmoid 也可以
        self.sigmoid = nn.Hardsigmoid()
        # self.sigmoid = nn.Sigmoid() # 原始版本

    def forward(self, x):
        # ----------------- 1. Shortcut 路径 -----------------
        shortcut_x = self.shortcut(x)

        # ----------------- 2. 主路径特征投影 -----------------
        # input_conv 是主要的特征变换，也是第一个残差连接的目标
        input_conv = self.conv_in(x)

        # ----------------- 3. 计算三个注意力流 -----------------
        # 局部流 (B, C, H, W)
        att_local = self.local_att(input_conv)

        # 全局流 (共享 MLP)
        att_global_avg = self.global_mlp(self.avg_pool(input_conv))
        att_global_max = self.global_mlp(self.max_pool(input_conv))

        # ----------------- 4. 融合与门控 -----------------
        # 融合三个流
        scale_logits = att_local + att_global_avg + att_global_max

        # 通过 sigmoid/hardsigmoid 得到注意力权重 (B, C, H, W)
        scale_weights = self.sigmoid(scale_logits)

        # 将权重应用于 conv_out 路径
        output_conv = self.conv_out(scale_weights)

        # ----------------- 5. 组合（残差连接） -----------------
        # 内部残差连接 (在变换后的空间)
        out = output_conv + input_conv

        # 外部（shortcut）残差连接 (与原始输入)
        out = out + shortcut_x

        return out


class MBFFN(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(MBFFN, self).__init__()
        inter_channels = int(in_channels // reduction)

        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        input_conv = self.conv_in(x)

        scale1 = self.local_att(input_conv)
        scale2 = self.global_att(input_conv)
        scale3 = self.global_att_max(input_conv)

        scale_out = scale1 + scale2 + scale3
        scale_out = self.sigmoid(scale_out)
        output_conv = self.conv_out(scale_out)
        out = output_conv + input_conv
        out = out + x
        return out


# 全局通道注意力
class GlobalChannelAttn(nn.Module):
    """
    Squeeze-and-Excitation 模块，用于注入全局通道注意力。
    输入: [B, C, N, 1]
    输出: [B, C, N, 1] (经过重标定的特征)
    """

    def __init__(self, channels, reduction=4):
        super(GlobalChannelAttn, self).__init__()
        # 确保中间通道数至少为1
        inter_channels = max(channels // reduction, 1)

        self.se_block = nn.Sequential(
            # Squeeze: [B, C, N, 1] -> [B, C, 1, 1]
            nn.AdaptiveAvgPool2d(1),

            # Excite: [B, C, 1, 1] -> [B, C_inter, 1, 1] -> [B, C, 1, 1]
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # global_scores: [B, C, 1, 1]
        global_scores = self.se_block(x)

        # Inject (注入): [B, C, N, 1] * [B, C, 1, 1] (利用广播)
        return x * global_scores


# 维度转换
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels

        self.use_short_cut = use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                # 不同点
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                # 不同点
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out


# TAG OA块
class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )

        # self.gcn = GCN_Block(points)
        # self.linear_0 = nn.Conv2d(points, 1, (1, 1))

        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # print(x.size(0))
        out = self.conv1(x)
        # print(out.size())
        # w0 = self.linear_0(out).view(x.size(0), -1) #w0[32,2000]
        out = out + self.conv2(out)  # + self.gcn(out, w0.detach())
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        # 卷积 bcn1
        embed = self.conv(x)  # b*k*n*1
        # 归一化 每个管道的值 bcn
        S = torch.softmax(embed, dim=2).squeeze(3)
        # X x Softmax
        # X(bcn) x bnc -> (bcc)->(bcc1)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class DGCNN_Layer(nn.Module):
    def __init__(self, knn_num=10, in_channel=128):
        super(DGCNN_Layer, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                # [32,128,2000,9]→[32,128,2000,3]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),  # [32,128,2000,3]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                # [32,128,2000,6]→[32,128,2000,2]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),  # [32,128,2000,2]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # print(self.knn_num)
        out = self.conv(x)  # BCN1
        return out


class AdaptiveSampling(nn.Module):
    def __init__(self, in_channel, num_sampling, use_bn=True):
        nn.Module.__init__(self)
        self.num_sampling = num_sampling
        if use_bn:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, num_sampling, kernel_size=1))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(in_channel, num_sampling, kernel_size=1))

    def forward(self, F, P):
        # F: BCN--->BCN1
        F = F.unsqueeze(3)
        weights = self.conv(F)  # BCN1
        W = torch.softmax(weights, dim=2).squeeze(3)  # BMN1
        # BCN @ BNM
        # 1 128 48
        F_M = torch.bmm(F.squeeze(3), W.transpose(1, 2))
        # 1 128 2000 9 x 1 48 128
        P_M = torch.bmm(P, W.transpose(1, 2))
        return F_M, P_M


class LearnableKernel(nn.Module):
    """
    Learnable Gaussian Kernel optimized for efficiency and numerical stability.

    输入:
      channels: 输入特征维度
      head: 注意力头数
      beta: 初始核尺度参数
      beta_learnable: 是否让 beta 可学习
      local_topk: 可选局部近邻数 (None 表示全局核)
    """

    def __init__(self, channels, head, beta, beta_learnable=True, local_topk=None):
        super().__init__()
        self.channels = channels
        self.head = head
        self.local_topk = local_topk  # 如果为None表示全局，否则保留topk邻域
        self.head_dim = channels // head

        # 两个线性映射层：位置embedding & value投影
        self.pos_filter = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.value_filter = nn.Conv1d(channels, channels, kernel_size=1)

        # beta 参数（高斯核带宽）
        if beta_learnable:
            # softplus参数化确保beta>0
            init_param = np.log(np.exp(beta) - 1.0 + 1e-6)
            self._beta_param = nn.Parameter(torch.tensor([init_param], dtype=torch.float32))
        else:
            self.register_buffer('_fixed_beta', torch.tensor([float(beta)], dtype=torch.float32))
            self._beta_param = None

    def _get_beta(self, device, dtype):
        if self._beta_param is not None:
            return F.softplus(self._beta_param).to(device=device, dtype=dtype)
        else:
            return self._fixed_beta.to(device=device, dtype=dtype)

    def forward(self, pos_bot, corr_feats):
        """
        pos_bot: [B, C, N]
        corr_feats: [B, C, N]
        返回:
            kernel: [B, N, N] (或局部稀疏核)
            equation_F: [B, N, C]
        """
        B, C, N = pos_bot.shape
        device, dtype = pos_bot.device, pos_bot.dtype

        # (1) 线性变换
        pos = self.pos_filter(pos_bot)  # [B, C/2, N]
        value = self.value_filter(corr_feats)  # [B, C, N]

        # (2) reshape 多头形式
        pos = pos.view(B, self.head, self.head_dim // 2, N)  # [B, H, Dp, N]
        value = value.view(B, self.head, self.head_dim, N)  # [B, H, Dv, N]

        # (3) 高斯核计算 (向量化, 无 torch.cdist)
        # ||pi - pj||^2 = pi^2 + pj^2 - 2 pi·pj
        pos_t = pos.transpose(-1, -2)  # [B, H, N, Dp]
        pos_sq = (pos_t ** 2).sum(-1, keepdim=True)  # [B, H, N, 1]
        dist2 = pos_sq + pos_sq.transpose(-1, -2) - 2 * torch.matmul(pos_t, pos_t.transpose(-1, -2))
        dist2 = dist2.clamp_min(0.0)  # 保证非负
        beta = self._get_beta(device, dtype)

        kernel = torch.exp(-beta * dist2)  # 高斯核 [B, H, N, N]

        # (4) 可选局部截断（sparse kernel）
        if self.local_topk is not None and self.local_topk < N:
            topk_vals, topk_idx = torch.topk(kernel, self.local_topk, dim=-1)
            mask = torch.zeros_like(kernel)
            mask.scatter_(-1, topk_idx, 1.0)
            kernel = kernel * mask  # 稀疏核

        # (5) 汇总多头结果（均值融合）
        kernel = kernel.mean(dim=1)  # [B, N, N]
        equation_F = value.mean(dim=1).transpose(1, 2).contiguous()  # [B, N, C]

        return kernel, equation_F


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1), \
            nn.Conv1d(channels, channels, kernel_size=1), \
            nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1),
            nn.BatchNorm1d(2 * channels), nn.ReLU(),
            nn.Conv1d(2 * channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1), \
            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1), \
            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim=-1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new


class DMFC_block(nn.Module):
    def __init__(self, channels, lamda, beta, layer, head, ker_head, num_sampling, lamda_learnable=True, use_bn=True):
        super().__init__()
        self.head = head
        self.ker_head = ker_head
        self.channels = channels
        self.num_sampling = num_sampling
        self.layer = layer
        self.beta = beta

        # 双路采样：分别对卷积特征和空间特征采样
        self.conv_sampling = AdaptiveSampling(channels, self.num_sampling)
        self.spatial_sampling = AdaptiveSampling(channels, self.num_sampling)

        # 双路注意力传播
        self.conv_to_spatial = AttentionPropagation(channels, self.head)
        self.spatial_to_conv = AttentionPropagation(channels, self.head)

        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )

        # 特征权重（可选）
        self.feats_weight = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, conv_feats, spatial_feats):
        """
        conv_feats: [B, C, N, 1] 卷积特征（丰富的语义信息）
        spatial_feats: [B, C, N, 1] 空间特征（位置、几何信息）
        return: 融合后的特征 [B, C, N, 1]
        """
        # 1. 分别采样
        conv_feats_N = conv_feats.squeeze(3)  # [B, C, N]
        spatial_feats_N = spatial_feats.squeeze(3)  # [B, C, N]

        # 卷积特征的代表点
        conv_repre, _ = self.conv_sampling(conv_feats_N, conv_feats_N)
        # 空间特征的代表点
        spatial_repre, _ = self.spatial_sampling(spatial_feats_N, spatial_feats_N)

        # 2. 双向注意力交互
        # 卷积特征关注空间结构
        conv_enhanced = self.conv_to_spatial(conv_feats_N, spatial_repre)
        # 空间特征关注语义信息
        spatial_enhanced = self.spatial_to_conv(spatial_feats_N, conv_repre)

        # 3. 特征融合
        fused_feats = self.feature_fusion(
            torch.cat([conv_enhanced, spatial_enhanced], dim=1)
        )

        # 4. 可选：应用特征权重
        weight = self.feats_weight(fused_feats)
        fused_feats = fused_feats * weight

        return fused_feats.unsqueeze(3)


class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        nn.Module.__init__(self)
        channels = net_channels
        # 层数
        self.layer_num = depth

        # l2 OAFilter块
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, data):
        # data: b*c*n*1

        x1_1 = data
        # 划分簇
        x_down = self.down1(x1_1)
        # OA
        x2 = self.l2(x_down)
        # 分解簇
        x_up = self.up1(x1_1, x2)

        out = torch.cat([x1_1, x_up], dim=1)
        return self.shot_cut(out)


class DSBlock(nn.Module):
    def __init__(self, net_channels, clusters=256, knn_num=6):
        nn.Module.__init__(self)
        channels = net_channels
        self.k_num = knn_num
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.DGCNN_MAX_Block = DGCNN_MAX_Block(self.k_num, l2_nums)

    def forward(self, data):
        # data: b*c*n*1

        x1_1 = data
        x_down = self.down1(x1_1).transpose(1, 2)
        x2 = self.DGCNN_MAX_Block(x_down).transpose(1, 2)

        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        return self.shot_cut(out)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # inner[32,2000,2000]内积？
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # distance[32,2000,2000]****记得回头看

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]


# NOTE 获取图像的特征点
def get_graph_feature(x, k=20, idx=None):
    # x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k)  # idx_out[32,2000,9]
    else:
        idx_out = idx

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base  # idx[32,2000,9] 把32个批次的标号连续了

    idx = idx.view(-1)  # idx[32*2000*9] 把32个批次连在一起了 [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # x[32,2000,128]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()  # feature[32,256,2000,9] 图特征
    return feature


# 构造一个获取特征的卷积层
class Conv_to_feature(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(Conv_to_feature, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 1)),  # [32,128,2000,9]→[32,128,2000,3]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),  # [32,128,2000,3]→[32,128,2000,1]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        # feature[32,128,2000,1]
        B, _, N, _ = features.shape
        # 获取高维特征映射结构为KxK，我们来制作knn块
        out = get_graph_feature(features, k=self.knn_num)
        # 卷积
        out = self.conv(out)  # out[32,128,2000,1]
        # 解压
        out = out.unsqueeze(3)
        return out


# NOTE 残差网络有助于解决梯度爆炸/消失的问题
class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4 [32,1,500,4] logits[32,2,500,1]
    mask = logits[:, 0, :, 0]  # [32,500] logits的第一层
    weights = logits[:, 1, :, 0]  # [32,500] logits的第二层

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        # k邻居块
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 1)),  # [32,128,2000,9]→[32,128,2000,3]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),  # [32,128,2000,3]→[32,128,2000,1]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        # feature[32,128,2000,1]
        B, _, N, _ = features.shape
        # 获取高维特征映射结构为KxK，我们来制作knn块
        out = get_graph_feature(features, k=self.knn_num)
        # 卷积
        out = self.conv(out)  # out[32,128,2000,1]
        # 获取最大值
        # [32,128,2000]
        out = out.max(dim=-1, keepdim=False)[0]
        # 解压
        out = out.unsqueeze(3)
        return out


# TAG GCN 权值结合，然后卷积
class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    # NOTE 激活并且切换
    def attention(self, w):
        # 双曲正切然后最后一排加空间
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)  # w[32,2000,1] 变成0到1的权重
        # wT x w
        A = torch.bmm(w.transpose(1, 2), w)  # A[32,1,1]
        return A

    # x与w的结合
    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()  # B=32,N=2000
        # 全局上下文嵌入fg = ()
        # 清空数据
        with torch.no_grad():
            A = self.attention(w)  # A[32,1,1]
            # 生成N*N单位矩阵
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()  # I[1,2000,2000]单位矩阵
            A = A + I  # A[32,2000,2000]
            D_out = torch.sum(A, dim=-1)  # D_out[32,2000]
            D = (1 / D_out) ** 0.5  # 权的倒数再开方
            # 对角矩阵
            D = torch.diag_embed(D)  # D[32,2000,2000]
            # DAD
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)  # L[32,2000,2000]
        # 交换成 BCNW->BNCW 每个单体的层次为单位计算
        # contiguous:确保矩阵在连续物理单元中
        out = x.squeeze(-1).transpose(1, 2).contiguous()  # out[32,2000,128]
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()  # out[32,128,2000,1]

        return out

    def forward(self, x, w):
        # x[32,128,2000,1],w[32,2000]
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        # 是否初始化
        self.initial = initial
        # 根据是否已经预测确定in channel为4或4
        self.in_channel = 4 if self.initial is True else 6
        # out channel
        self.out_channel = out_channel
        # 层数
        self.k_num = k_num
        # 是否预测
        self.predict = predict
        # 学习率
        self.sr = sampling_rate
        # self.lamda = config.lamda

        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        # NOTE GGCE：全局图上下文的聚合
        # gcn 入口为out_channel
        self.gcn = GCN_Block(self.out_channel)

        self.geom_embed = nn.Sequential(nn.Conv2d(4, self.out_channel, 1), \
                                        PointCN(self.out_channel))

        # def __init__(self, channels, lamda, beta, layer, head, ker_head, num_sampling, lamda_learnable=True, use_bn=True)
        self.dmfc = DMFC_block(self.out_channel, lamda=8, beta=0.1, layer=1, head=4, ker_head=1, num_sampling=48,
                               lamda_learnable=True, use_bn=True)




        # 2*ResNet+DGCNN_MAX_Block+2*ResNet+OABlock+2*ResNet
        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MBFFN(self.out_channel,self.out_channel),
            # NGCE
            DGCNN_MAX_Block(int(self.k_num * 2),self.out_channel),
            MBFFN(self.out_channel, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MBFFN(self.out_channel, self.out_channel),
            # CGCE
            OABlock(self.out_channel, clusters=256),
            MBFFN(self.out_channel, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        # 2*ResNet+DGCNN_MAX_Block+2*ResNet+OABlock+2*ResNet
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MBFFN(self.out_channel, self.out_channel),
            OABlock(self.out_channel, clusters=128),
            MBFFN(self.out_channel, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MBFFN(self.out_channel, self.out_channel),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            MBFFN(self.out_channel, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))


        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, outs=[], features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]  # indices[32,1000]剪枝剪掉一半
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)  # y_out 剪枝后保留的标签[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices)  # w_out 剪枝后保留的w0[32,1000]
        indices = indices.view(B, 1, -1, 1)  # indices[32,1,1000,1]

        outs_p = []

        if predict == False:
            with torch.no_grad():
                # out(BCN1)
                for out in outs:
                    # indices1 = indices.expend(-1, out.size(1), -1, -1)
                    indices1 = indices.repeat(1, out.size(1), 1, 1)
                    out = torch.gather(out, dim=2, index=indices1)  # x_out [32, 1, M, 4]
                    outs_p.append(out)
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))  # x_out 剪枝后保留的x[32,1,1000,4]
            return x_out, y_out, w_out, outs_p
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))  # x_out 剪枝后保留的x[32,1,500,4]
            feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, 128, 1, 1))  # feature_out 剪枝后保留的features[32,128,500,1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y, outs=None):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        if self.initial:
            outs_to_collect = []

        # TAG Step1 初始特征化
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  # contiguous断开out与x的依赖关系。out[32,4或6,2000,1]
        x1, x2 = out[:, :2, :, :], out[:, 2:4, :, :]
        out = self.conv(out)

        # TAG Step2 创建专门的特征化
        # --- 头部“温故” (Head "Reviewing the Old")---
        out_pos_rich = out
        if not self.initial and outs and len(outs) > 0:
            # outs[0] 是 ds_0 的早期局部/簇级特征
            out_pos_rich = outs[0]

        # --- “温故”结束 ---

        # --- 3. 创建 out_m (运动特征) ---
        motion = torch.cat((x1, x2 - x1), dim=1)
        # 运动卷积 [32 128 2000 1]
        out_m = self.geom_embed(motion)

        # --- 4. DMFC 修正 (使用丰富的 pos) ---
        # 注意：不要将结果存回 'out' 或 'out_pos'
        corrected_out_m = self.dmfc(out_pos_rich, out_m)  # [B, 128, N, 1]

        #TAG Step3 融合 (关键的修复)
        # 将丰富的上下文(out_pos_rich)和修正后的运动(corrected_out_m)融合
        out_fused = corrected_out_m+ out

        # TAG Step4 主干网络
        out = self.embed_0(out_fused)  # NGCE+CGCE=ResNet * 3 + DGCNN_MAX + ResNet * 3 + OABlock + ResNet * 3

        if self.initial:
            outs_to_collect.append(out)  # 收集 ds_0 的特征


        w0 = self.linear_0(out).view(B, -1)  # w0[32,2000]

        # TAG Step 5: 全局上下文 (GCN & Fusion)
        out_g = self.gcn(out, w0.detach())  # GGCE

        # 将簇级和全局的相加
        # TODO (将 out_g + out 替换为 out_g + out_fused，如果需要的话)
        out = out_g + out


        if self.initial:
            outs_to_collect.append(out)  # 收集 ds_0 的特征

        out = self.embed_1(out)  # CGCE+NGCE=ResNet * 2 + OABlock + ResNet * 2 + DGCNN_MAX + ResNet * 2

        if self.initial:
            outs_to_collect.append(out)  # 收集 ds_0 的特征



        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]
        # TAG 结束

        if self.predict == False:  # 剪枝，不预测
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1排序,w1_ds[32,2000],indices[32,2000]是索引
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,1000]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds, outs_p = self.down_sampling(x, y, w0, indices, outs_to_collect, None, self.predict)
            # print(outs_p[0].shape)
            # x_ds[32,1,1000,4],y_ds[32,1000],w0_ds[32,1000],ds：剪枝后？
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], outs_p
        else:  # 剪枝，出预测结果
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1排序,w1_ds[32,1000],indices[32,1000]是索引
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,500]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, [], out, self.predict)
            # x_ds[32,1,500,4],y_ds[32,500],w0_ds[32,500],out[32,128,500,1]也是剪枝后,ds：剪枝后？
            out = self.embed_2(out)
            w2 = self.linear_2(out)  # [32,2,500,1]
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class CLNet(nn.Module):
    def __init__(self, config):
        super(CLNet, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr)  # sampling_rate=0.5
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0, out_p = self.ds_0(x, y)  # 返回的是x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  # x_[32,1,1000,6] 剪枝后的特征并带上了权重信息

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1, out_p)  # x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat对称极线距离
        # print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

