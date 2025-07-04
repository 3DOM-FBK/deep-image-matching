import torch
import torch.nn as nn
import copy


def elu_feature_map(x):
    return nn.functional.elu(x) + 1

class AttentionPropagation(nn.Module):
    def __init__(self, channels, head, mode='full'):
        nn.Module.__init__(self)
        self.head = head
        self.mode = mode
        self.head_dim = channels // head
        if mode=='linear':
            self.feature_map = elu_feature_map
            self.eps = 1e-6

        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(inplace=True),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, x1, x2, kv_mask=None):
        # x1(q) attend to x2(k,v)
        batch_size = x1.shape[0]
        query, key, value = self.query_filter(x1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(x2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(x2).view(batch_size, self.head, self.head_dim, -1)

        if self.mode == 'full':
            QK = torch.einsum('bhdn,bhdm->bhnm', query, key)
            # set masked position to -1e6
            if kv_mask is not None:
                QK.masked_fill_(~(kv_mask[:, None, None, :]), float(-1e6))
            score = torch.softmax(QK / self.head_dim ** 0.5, dim = -1) # BHNM
            add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
            # assign_mat = torch.mean(torch.softmax(QK/self.head_dim**0.5,dim=-2),dim=1,keepdim=False).permute(0,2,1) # BMN
        elif self.mode == 'linear':
            # set masked position to zero
            if kv_mask is not None:
                key = key * kv_mask[:, None, None, :]
                value = value * kv_mask[:, None, None, :]
            Q = self.feature_map(query) # BHDN
            K = self.feature_map(key) # BHDM
            v_length = value.shape[-1] # BHVM
            value = value / v_length  # prevent fp16 overflow
            KV = torch.einsum("bhdm,bhvm->bhdv", K, value)
            Z = 1 / (torch.einsum("bhdn,bhd->bhn", Q, K.sum(dim=-1)) + self.eps)
            add_value = torch.einsum("bhdn,bhdv,bhn->bhvn", Q, KV, Z).reshape(batch_size, self.head_dim * self.head, -1) * v_length # B(HD)N
        else:
            raise KeyError

        add_value = self.mh_filter(add_value)
        x1_new = x1 + self.cat_filter(torch.cat([x1, add_value], dim=1))
        return x1_new


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class LayerBlock(nn.Module):
    def __init__(self, channels, head, layer_names, mode='full'):
        nn.Module.__init__(self)
        self.layer_names = layer_names
        cluster_layer = AttentionPropagation(channels, head, mode=mode)
        self.cluster_layers = nn.ModuleList([copy.deepcopy(cluster_layer) for _ in range(len(layer_names))])
        self.inlier_pre = InlinerPredictor(channels)
            
    def forward(self, xs, d, feat_piece, logits=None, loss=False):
        # xs: B1N4, d: BCN, feat_piece: BCP, logits: BN
        d_old = d.clone()
        if logits is not None:
            weights = torch.relu(torch.tanh(logits)) # BN
            mask = weights>0 # BN
        else:
            mask = None
        for layer, name in zip(self.cluster_layers, self.layer_names):
            if name == 'cluster':
                feat_piece = layer(feat_piece, d, mask) # BCP, BNP
            elif name == 'context':
                feat_piece = layer(feat_piece, feat_piece) # BCP
            elif name == 'decluster':
                d = layer(d, feat_piece) # BCN, BPN
            else:
                raise KeyError
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d-d_old), 1) # BN
        if loss:
            e_hat = weighted_8points(xs, logits)
        else:
            e_hat = None
        return d, logits, e_hat


class DeMatch(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num

        self.piece_tokens = nn.Parameter(torch.randn(config.net_channels, config.piece_num)) # CP
        self.register_parameter('piece_tokens', self.piece_tokens)
        self.pos_embed = PositionEncoder(config.net_channels)
        self.init_project = InitProject(config.net_channels)
        self.layer_blocks = nn.Sequential(
            *[LayerBlock(config.net_channels, config.head, config.layer_names, mode=config.attention_mode) for _ in range(self.layer_num)]
        )

    def forward(self, data, training=False):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1,3).squeeze(3) # B4N
        x1, x2 = input[:,:2,:], input[:,2:,:]
        motion = x2 - x1 # B2N

        pos = x1 # B2N
        pos_embed = self.pos_embed(pos) # BCN

        d = self.init_project(motion) + pos_embed # BCN
        feat_piece = self.piece_tokens.unsqueeze(0).repeat(batch_size, 1, 1) # CP->BCP

        res_logits, res_e_hat = [], []
        logits = None

        for i in range(self.layer_num):
            if i<self.layer_num-1:
                d, logits, e_hat = self.layer_blocks[i](data['xs'], d, feat_piece, logits=logits, loss=training) # BCN
                res_logits.append(logits), res_e_hat.append(e_hat)
            else:
                d, logits, e_hat = self.layer_blocks[i](data['xs'], d, feat_piece, logits=logits, loss=True) # BCN
                res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

