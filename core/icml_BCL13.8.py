import torch
import torch.nn as nn
from loss import batch_episym
import torch.nn.functional as F

'''
@Title: icml_BCL13.8
@Author: ChuyongWei
@Date: 2025-09-24
@Version: 2.0
@Description:
尝试在加入NGCE+CGCE之后加入motion融入BCRBlock

4.加入att模块将前面max和out进行融合
5.由于4发生了过拟合经过检查发现aff和bcr的代码相似，是不正常的现象因此做修改，将att的思路带入bcr
8.这次加入gca模块
@Evaluation
map5 71.08
4.map5 70.43
8.map5 71.375
原文中这块方法原本是用于`运动`的注入方式但是现在修改为了用于隐式和显式的特征融合
？？？
我看BCL是用的隐式和显式的，不过就是说如果使用了显式和隐式的方法的话就显的重复了

'''


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


# 维度转换
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class CPT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPT, self).__init__()

        self.graph1_conv = nn.Conv2d(2, in_channels, kernel_size=1)
        self.graph2_conv = nn.Conv2d(2, in_channels, kernel_size=1)

        self.q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.temperature = torch.sqrt(torch.tensor(in_channels))
        self.temperature2 = torch.sqrt(torch.tensor(in_channels))

    def forward(self, PointCN1, x):
        q = self.q(PointCN1).squeeze(3)
        k = self.k(PointCN1).squeeze(3)
        v = self.v(PointCN1).squeeze(3)

        # NOTE Geometric Semantic Extraction (GSE) 几何语义提取
        # 当整合予以信息信息确保了一个清晰并且强健的图像对关系的编码
        # x = x.transpose(1, 3).contiguous()
        graph_1_coordinates = x[:, :2, :, :]  # 形状: [B, 2, N, 1]
        graph_2_coordinates = x[:, 2:4, :, :]  # 形状: [B, 2, N, 1]
        graph_1 = self.graph1_conv(graph_1_coordinates)
        graph_2 = self.graph2_conv(graph_2_coordinates)
        graph_context = graph_1 + graph_2
        graph_context = graph_context.squeeze(3)  # B2N1

        # GCP = Q/t@gT
        #  ATT = Q/t @ KT
        # BCN@BNC 计算邻居点的相似度
        #
        graph_context_position = torch.matmul(q / self.temperature2, graph_context.transpose(1, 2))
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        # NOTE  Geometric Semantic Feature Fusion (GSFF) - 几何语义特征融合
        attn = attn + graph_context_position
        # attn = self.dropout(F.softmax(attn, dim=-1))
        #  SoftMax功能使相关矩阵归一化以产生注意力权重，从而有效地捕获了特征点之间的长期依赖性。
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).unsqueeze(3)
        return output


# Multi-Branch Feed Forward Network, MBFFN
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


class CGA_Module(nn.Module):
    def __init__(self, channels):
        super(CGA_Module, self).__init__()
        self.CPA = CPT(channels, channels)
        self.LayerNorm1 = nn.LayerNorm(channels, eps=1e-6)
        self.MBFFN = MBFFN(channels, channels)
        self.LayerNorm2 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, feature, Position_feature):
        # out x
        # CPT
        CPT_feature = self.CPA(feature, Position_feature)
        #
        CPT_feature = CPT_feature + feature
        # LayerNorm 1
        CPT_feature_LN1 = CPT_feature.squeeze(3).transpose(-1, -2)
        CPT_feature_LN1 = self.LayerNorm1(CPT_feature_LN1)
        CPT_feature_LN1 = CPT_feature_LN1.transpose(-1, -2).unsqueeze(3)
        # MBFFN
        MBFFN_feature = self.MBFFN(CPT_feature_LN1)
        # LayerNorm 2
        MBFFN_feature_LN2 = MBFFN_feature.squeeze(3).transpose(-1, -2)
        MBFFN_feature_LN2 = self.LayerNorm2(MBFFN_feature_LN2)
        MBFFN_feature_LN2 = MBFFN_feature_LN2.transpose(-1, -2).unsqueeze(3)
        # out
        out = CPT_feature_LN1 + MBFFN_feature_LN2

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
    # 返回(values, indices)
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


# FIX 修改前
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


# TAG 计算权重
# NOTE 计算全局我们当然要使用到权重
class BEBlock(nn.Module):
    def __init__(self, channels, r=4, k_num=8):
        super(BEBlock, self).__init__()
        self.k_num = k_num
        inter_channels = int(channels // r)

        self.project_be = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.project_knn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # NOTE 计算均值的话。。我们就能看到全局的影响
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # b*128*1*1
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att = DGCNN_Layer(self.k_num, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取特征
        x1 = self.project_be(x)

        # 获取局部的特征
        x_local = self.project_knn(x1)
        x_local = self.local_att(get_graph_feature(x_local, k=self.k_num))

        # 平均池全局搜索
        # 然后加上局部特征就是我们的样本了，使用sigmoid函数获取权重
        xlg = self.global_att(x1) + x_local
        weight = self.sigmoid(xlg)
        return weight


# TAG 全局获取模块
class BCRBlock(nn.Module):
    def __init__(self, channels, k_num=8, r=4):
        super(BCRBlock, self).__init__()
        self.channels = channels
        self.ratio = r
        self.k = k_num
        self.Weight = BEBlock(channels, self.ratio, self.k)
        self.resnet_1 = ResNet_Block(self.channels, self.channels, pre=True)
        self.resnet_12 = ResNet_Block(self.channels * 2, self.channels, pre=True)
        self.resnet_2 = nn.Sequential(
            ResNet_Block(self.channels, self.channels, pre=False),
            ResNet_Block(self.channels, self.channels, pre=False),
            ResNet_Block(self.channels, self.channels, pre=False)
        )

    def forward(self, x, residual):
        x_1 = x + residual
        x_1 = self.resnet_1(x_1)
        wei = self.Weight(x_1)
        x = 2 * x * wei + 2 * residual * (1 - wei)
        x = self.resnet_2(x)
        return x


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

    # NOTE 双曲正切激活并且内积和将信息聚合到一起
    def attention(self, w):
        # BN
        # 双曲正切然后最后一排加空间
        # BN1
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)  # w[32,2000,1] 变成0到1的权重
        # wT x w
        # B1N x BN1 ->B11
        # 特征内积和聚合到一个数值
        # NOTE
        A = torch.bmm(w.transpose(1, 2), w)  # A[32,1,1]
        return A

    '''
    x与w的结合
    w处理
    + A：w双曲正切内积和
    + A+I求每行的和取倒数然后开方换成对角矩阵得D
    + L = DAD (BNN)
    结合
    + x(BCN1)->BNC
    + out=L(BNN)@X(BNC)->BNC->BCN1
    '''

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
        # L(BNN) @ X(BNC)->BNC->BNC1
        out = torch.bmm(L, out).unsqueeze(-1)
        # BCN1
        out = out.transpose(1, 2).contiguous()  # out[32,128,2000,1]

        return out

    def forward(self, x, w):
        # x[32,128,2000,1],w[32,2000]
        # NOTE 将我们处理后得特征点和权重进行结合
        out = self.graph_aggregation(x, w)
        # 卷积
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

        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.geom_embed = nn.Sequential(nn.Conv2d(4, self.out_channel, 1), \
                                        PointCN(self.out_channel))

        # XXX 这边可以考虑加入我们的学习率
        # BCRBlock(out_channel, self.k_num, self.sr)
        self.bcr = BCRBlock(out_channel, self.k_num)
        self.bcr2 = BCRBlock(out_channel, self.k_num)

        self.cpa = CGA_Module(out_channel)
        # self.aff = AFF(out_channel)

        # gcn 入口为out_channel
        self.gcn = GCN_Block(self.out_channel)

        # 2*ResNet+DGCNN_MAX_Block+OABlock+
        self.embed_00 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # NGCE
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_01 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # CGCE
            OABlock(self.out_channel, clusters=256),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]  # indices[32,1000]剪枝剪掉一半
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)  # y_out 剪枝后保留的标签[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices)  # w_out 剪枝后保留的w0[32,1000]
        indices = indices.view(B, 1, -1, 1)  # indices[32,1,1000,1]

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))  # x_out 剪枝后保留的x[32,1,1000,4]
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))  # x_out 剪枝后保留的x[32,1,500,4]
            feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, 128, 1, 1))  # feature_out 剪枝后保留的features[32,128,500,1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  # contiguous断开out与x的依赖关系。out[32,4或6,2000,1]
        x1, x2 = out[:, :2, :, :], out[:, 2:4, :, :]
        out = self.conv(out)  # out[32,128,2000,1]

        motion = torch.cat((x1, x2 - x1), dim=1)
        out_m = self.geom_embed(motion)

        # TAG 开始
        ## NOTE 局部领域上下图+ 簇级图上下文
        out_max = self.embed_00(out)  # NGCE+CGCE=ResNet * 3 + DGCNN_MAX + ResNet * 3 + OABlock + ResNet * 3
        out = self.cpa(out,out_max)
        out = self.embed_01(out_max)
        # FIX BCRBlock
        out = self.bcr(out, out_m)

        w0 = self.linear_0(out).view(B, -1)  # w0[32,2000]

        # NOTE 全局图上下文
        out_g = self.gcn(out, w0.detach())  # GGCE

        # 将簇级和全局的相加
        out = out_g + out

        # out = self.bcr2(torch.cat([out, out_g], dim=1))
        out = self.embed_1(out)  # CGCE+NGCE=ResNet * 2 + OABlock + ResNet * 2 + DGCNN_MAX + ResNet * 2
        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]
        # TAG 结束

        if self.predict == False:  # 剪枝，不预测
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1排序,w1_ds[32,2000],indices[32,2000]是索引
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,1000]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            # x_ds[32,1,1000,4],y_ds[32,1000],w0_ds[32,1000],ds：剪枝后？
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:  # 剪枝，出预测结果
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1排序,w1_ds[32,1000],indices[32,1000]是索引
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,500]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
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

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)  # 返回的是x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)  # 变成0到1的权重[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  # x_[32,1,1000,6] 剪枝后的特征并带上了权重信息

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)  # x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat对称极线距离
        # print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

