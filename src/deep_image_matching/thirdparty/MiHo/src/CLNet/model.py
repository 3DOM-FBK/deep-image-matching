import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)

    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))

    return ys

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
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

def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

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
    _, v = torch.linalg.eigh(XwX)

    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)
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
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w, w.transpose(1, 2))
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
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
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN_Block(self.out_channel)

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_Block(self.k_num, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)

        out = self.embed_0(out)
        w0 = self.linear_0(out).view(B, -1)

        out_g = self.gcn(out, w0.detach())
        out = out_g + out

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class CLNet(nn.Module):
    def __init__(self, config):
        super(CLNet, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)#1.0)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat
