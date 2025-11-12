import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange

'''
@Title: MSGA2
@Author: ChuyongWei
@Date: 2025-07-22
@Version: 1.2
@Description:
去除上下的OANet

'''


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


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
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
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
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, data):
        # data: b*c*n*1

        x1_1 = data
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
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
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)  # out[32,128,2000,1]
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        return out


# TAG MSFormer 注意力机制
# NOTE 多头注意力,将数据conv扩展3C分成KQV三层,然后多头化
# attn = KQ
# 分级多种注意力*各自softmax
#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))  # QKV 卷积 + 深度卷积
        q, k, v = qkv.chunk(3, dim=1)  # 拆成三份：Q, K, V
        # 多头注意力+L2归一化
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        # 制作0矩阵
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        # BHCC*w(H)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 最稀疏的注意力（保留最强的50%）
        # NOTE 获取Top-k的索引值
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        # 把 index 对应位置的值设为 1.0，其余保持为 0.0，用来标记“哪些位置是 Top-K 的注意力得分”。
        # tensor.scatter_(dim, index, src)
        # 在 dim 维度上，将 src 中的值根据 index 指定的位置写入 tensor。
        mask1.scatter_(-1, index, 1.)
        #  非top-k位置设为 -inf
        # torch.where(condition, a, b)
        # 等同于output = a if condition else b
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        # 稍密集
        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        # 更密集
        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))
        # 最密集的注意力
        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # softMax BHCC
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)
        # BHCC @ BHC(HW)->BHC(HW)
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)
        # 计算加权得分
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        # 回来
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 给分
        out = self.project_out(out)
        return out


# TAG AFF:Attentional feature fusion Squeeze-and-Excitation改进版
# 融合了 局部注意力 和 全局注意力，来动态地调整通道特征的权重。我们来逐行解析这段 SE_Block 的逻辑和关键点。
# NOTE 该模块选择性地保留了有效的图形上下文。
class SE_Block(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        # 首先使用 全局平均池化 提取每个通道的全局上下文表示。
        # 然后通过两个 3x3 卷积层+BN+ReLU 进行非线性映射，学习全局注意力权重。
        # 输出 shape 为 (B, C, 1, 1)，表示每个通道的“全局重要性”。
        self.global_att = nn.Sequential(
            # NOTE 全局的秘密
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        # 使用标准的 3x3 卷积提取 局部特征上下文。
        # 没有池化操作，保留了原始空间结构 (B, C, H, W)。
        # 相当于一个轻量的通道注意力模块，能感知局部特征的强度或模式。
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 局部注意力
        # BCHW
        xl = self.local_att(x)
        # 全局注意力
        # BCHW
        xg = self.global_att(x)
        # 逐通道融合
        xlg = xl * xg
        # 计算得分
        wei = self.sigmoid(xlg)

        out = wei * x
        return out


# TAG MSFormer Topk_ATT+SE的块
# NOTE 使用BN(x)做输入是一个不错的方法
class Topk_transformer(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.attn = Attention(channels, num_heads=4)
        self.se_block = SE_Block(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # BCGG
        # BN(x)
        x_norm = self.norm1(x)
        # BN(x+ SE(x_norm)+ATT(x_norm))
        # BCGG
        x_attn = self.attn(x_norm)
        # BCGG
        x_se = self.se_block(x_norm)
        x = x + x_se + x_attn
        x_norm = self.norm2(x)
        # CBRC(c/4)
        # BCGG
        x = x + self.ffn(x_norm)
        return x


class LayerBlock(nn.Module):
    def __init__(self, channels, grid=16):
        nn.Module.__init__(self)
        self.output = grid * grid
        self.upsample = diff_pool(channels, self.output)
        self.grid_num = grid
        self.glo_exchange1 = Topk_transformer(channels)
        self.glo_exchange2 = Topk_transformer(channels)
        self.downsample = diff_unpool(channels, self.output)

    def forward(self, x):
        # BCN1
        # 分割簇 BCO
        x_up = self.upsample(x)
        # BCGG
        x_up = x_up.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        #
        x_glo = self.glo_exchange1(x_up)
        x_glo = self.glo_exchange2(x_glo)
        # BCO
        x_glo = x_glo.reshape(x.shape[0], x.shape[1], self.grid_num * self.grid_num)
        #
        x = self.downsample(x, x_glo)
        return x


class GloBlock(nn.Module):
    def __init__(self, channels, k_num, grid=16):
        nn.Module.__init__(self)
        self.transformer1 = LayerBlock(channels, grid)

    def forward(self, x):
        out = self.transformer1(x)
        return out


class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)  # w[32,2000,1] 变成0到1的权重
        A = torch.bmm(w.transpose(1, 2), w)  # A[32,1,1]
        return A  # TODO zfy

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()  # B=32,N=2000
        with torch.no_grad():
            A = self.attention(w)  # A[32,1,1]
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()  # I[1,2000,2000]单位矩阵
            A = A + I  # A[32,2000,2000]    TODO zfy
            D_out = torch.sum(A, dim=-1)  # D_out[32,2000]
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)  # D[32,2000,2000]
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)  # L[32,2000,2000]
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
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        # self.gcn = GCN_Block(self.out_channel)
        self.glo_block = GloBlock(self.out_channel, k_num=self.k_num)
        # self.embed_0 = nn.Sequential(
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     DGCNN_MAX_Block(int(self.k_num * 2),self.out_channel),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     OABlock(self.out_channel, clusters=256),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        # )
        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # FIX 修改
            # ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # OABlock(self.out_channel, clusters=256),
            # ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        # self.embed_1 = nn.Sequential(
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     DGCNN_MAX_Block(int(self.k_num * 2),self.out_channel),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        #     ResNet_Block(self.out_channel, self.out_channel, pre=False),
        # )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # FIX 修改
            # ResNet_Block(self.out_channel, self.out_channel, pre=False),
            # OABlock(self.out_channel, clusters=128),
            # ResNet_Block(self.out_channel, self.out_channel, pre=False),
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
        out = self.conv(out)  # out[32,128,2000,1]

        out = self.embed_0(out)  # NGCE+CGCE=ResNet * 3 + DGCNN_MAX + ResNet * 3 + OABlock + ResNet * 3
        w0 = self.linear_0(out).view(B, -1)  # w0[32,2000]

        # FIX
        out_g = self.glo_block(out)
        # out_g = self.gcn(out, w0.detach())  # GGCE
        out = out_g + out

        out = self.embed_1(out)  # CGCE+NGCE=ResNet * 2 + OABlock + ResNet * 2 + DGCNN_MAX + ResNet * 2
        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]

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

