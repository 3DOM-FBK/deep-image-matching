import torch
import torch.nn as nn
from loss import batch_episym

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x) 
    xx = torch.sum(x**2, dim=1, keepdim=True) 
    pairwise_distance = -xx - inner - xx.transpose(2, 1) 

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   

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

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze())
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
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

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

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
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
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
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
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
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out
    

class GS(nn.Module):
    def __init__(self, in_channel):
        super(GS, self).__init__()
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
        out = x + out
        return out

class SCE_Layer(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(SCE_Layer, self).__init__()
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

    def forward(self, x):
        out = self.conv(x) 
        return out

class GCN(nn.Module):
    def __init__(self, in_channel):
        super(GCN, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def gcn(self, x, w):
        B, _, N, _ = x.size() 

        with torch.no_grad():
            w = torch.relu(torch.tanh(w)).unsqueeze(-1) 
            A = torch.bmm(w, w.transpose(1, 2))
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
        out = self.gcn(x, w)
        out = self.conv(out)
        return out

class CCI_Layer(nn.Module):
    def __init__(self, in_channel):
        nn.Module.__init__(self)

        self.attq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, n1, n2, n3):
        q1 = self.attq(n1).squeeze(3)
        k1 = self.attk(n2).squeeze(3)
        v1 = self.attv(n3).squeeze(3)
        scores = torch.bmm(q1.transpose(1, 2), k1)
        att = torch.softmax(scores, dim=2)
        out = torch.bmm(v1, att.transpose(1, 2))
        out = out.unsqueeze(3)
        out = self.conv(out)
        out = n3 + self.gamma * out
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
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), #4»ò6 ¡ú 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn1 = GCN(self.out_channel)
        self.gcn2 = GCN(self.out_channel)
          
        self.sce1 = SCE_Layer(self.k_num, self.out_channel)
        self.sce2 = SCE_Layer(self.k_num, self.out_channel)
        self.sce3 = SCE_Layer(self.k_num, self.out_channel)

        self.cci1 = CCI_Layer(self.out_channel)
        self.cci2 = CCI_Layer(self.out_channel)
        self.cci3 = CCI_Layer(self.out_channel)
        
        self.sce11 = SCE_Layer(self.k_num, self.out_channel)
        self.sce22 = SCE_Layer(self.k_num, self.out_channel)
        self.sce33 = SCE_Layer(self.k_num, self.out_channel)

        self.cci11 = CCI_Layer(self.out_channel)
        self.cci22 = CCI_Layer(self.out_channel)
        self.cci33 = CCI_Layer(self.out_channel)
        
        self.down1 = diff_pool(self.out_channel, 250)
        self.l1 = []
        for _ in range(2):
            self.l1.append(OAFilter(self.out_channel, 250))
        self.up1 = diff_unpool(self.out_channel, 250)
        self.l1 = nn.Sequential(*self.l1)
        
        self.down2 = diff_pool(self.out_channel, 250)
        self.l2 = []
        for _ in range(2):
            self.l2.append(OAFilter(self.out_channel, 250))
        self.up2 = diff_unpool(self.out_channel, 250)
        self.l2 = nn.Sequential(*self.l2)

        self.embed_00 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.embed_01 = nn.Sequential(
            ResNet_Block(self.out_channel *4, self.out_channel, pre=True),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        
        self.embed_02 = nn.Sequential(
            ResNet_Block(self.out_channel *4, self.out_channel, pre=True),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.gs = GS(self.out_channel)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.mlp1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.mlp2 = nn.Conv2d(self.out_channel, 1, (1, 1))
        
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

        idx_sn = knn(out.squeeze(-1), k=self.k_num)

        out = self.conv(out) 

        out = self.embed_00(out) 
        idx_fn1 = knn(out.squeeze(-1), k=self.k_num)

        w_p1 = self.mlp1(out).view(B, -1)
        out_gs1 = self.gcn1(out, w_p1.detach())
        idx_gn1 = knn(out_gs1.squeeze(-1), k=self.k_num)

        C_S1 = self.sce1(get_graph_feature(out, k=self.k_num, idx= idx_sn))
        C_F1 = self.sce2(get_graph_feature(out, k=self.k_num, idx= idx_fn1))
        C_G1 = self.sce3(get_graph_feature(out, k=self.k_num, idx= idx_gn1))

        I_S1 = self.cci1(C_F1, C_G1, C_S1)
        I_F1 = self.cci2(C_G1, C_S1, C_F1)
        I_G1 = self.cci3(C_S1, C_F1, C_G1)
        
        x_down = self.down1(out)
        x2 = self.l1(x_down)
        x_up = self.up1(out, x2)

        out = self.embed_01(torch.cat((I_S1, I_F1, I_G1, x_up), 1))

        idx_fn2 = knn(out.squeeze(-1), k=self.k_num)
                
        w_p2 = self.mlp2(out).view(B, -1)
        out_gs2 = self.gcn2(out, w_p2.detach())
        idx_gn2 = knn(out_gs2.squeeze(-1), k=self.k_num)

        C_S2 = self.sce11(get_graph_feature(out, k=self.k_num, idx= idx_sn))
        C_F2 = self.sce22(get_graph_feature(out, k=self.k_num, idx= idx_fn2))
        C_G2 = self.sce33(get_graph_feature(out, k=self.k_num, idx= idx_gn2))

        I_S2 = self.cci11(C_F2, C_G2, C_S2)
        I_F2 = self.cci22(C_G2, C_S2, C_F2)
        I_G2 = self.cci33(C_S2, C_F2, C_G2)
        
        x_down = self.down2(out)
        x2 = self.l2(x_down)
        x_up = self.up2(out, x2)

        out = self.embed_02(torch.cat((I_S2, I_F2, I_G2, x_up), 1))
        
        w0 = self.linear_0(out).view(B, -1)  
        out = self.gs(out, w0.detach()) 

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

class NCMNet(nn.Module):
    def __init__(self, config):
        super(NCMNet, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)
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

