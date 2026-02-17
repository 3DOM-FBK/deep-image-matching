from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def MLP_no_ReLU(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Encoding of geometric properties using MLP """
    def __init__(self, keypoint_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([keypoint_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, kpts):
        if self.use_dropout:
            return self.dropout(self.encoder(kpts))
        return self.encoder(kpts)

class NormalEncoder(nn.Module):
    """ Encoding of geometric properties using MLP """
    def __init__(self, normal_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP_no_ReLU([normal_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, kpts):
        if self.use_dropout:
            return self.dropout(self.encoder(kpts))
        return self.encoder(kpts)


class DescriptorEncoder(nn.Module):
    """ Encoding of visual descriptor using MLP """
    def __init__(self, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([feature_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, descs):
        residual = descs
        if self.use_dropout:
            return residual + self.dropout(self.encoder(descs))
        return residual + self.encoder(descs)


class AFTAttention(nn.Module):
    """ Attention-free attention """
    def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.dim = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # q = torch.sigmoid(q)
        k = k.T
        k = torch.softmax(k, dim=-1)
        k = k.T
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv
        x = self.proj(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        # x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
        # self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        # x = self.layer_norm(x)
        return x


class AttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1):
        super().__init__()
        self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
        self.ffn = PositionwiseFeedForward(feature_dim, dropout=dropout, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        x = self.attn(x)
        x = self.ffn(x)
        return x


class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, layer_num: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalLayer(feature_dim, dropout=dropout, p=p)
            for _ in range(layer_num)])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            desc = layer(desc)
        return desc


class FeatureBooster(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128],
        'Attentional_layers': 3,
        'last_activation': 'relu',
        'l2_normalization': True,
        'output_dim': 128
    }

    def __init__(self, config, dropout=False, p=0.1, use_kenc=True, use_normal=True, use_cross=True):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.use_kenc = use_kenc
        self.use_cross = use_cross
        self.use_normal = use_normal

        if use_kenc:
            self.kenc = KeypointEncoder(self.config['keypoint_dim'], self.config['descriptor_dim'], self.config['keypoint_encoder'], dropout=dropout)

        if use_normal:
            self.nenc = NormalEncoder(self.config['normal_dim'], self.config['descriptor_dim'], self.config['normal_encoder'], dropout=dropout)

        if self.config.get('descriptor_encoder', False):
            self.denc = DescriptorEncoder(self.config['descriptor_dim'], self.config['descriptor_encoder'], dropout=dropout)
        else:
            self.denc = None

        if self.use_cross:
            self.attn_proj = AttentionalNN(feature_dim=self.config['descriptor_dim'], layer_num=self.config['Attentional_layers'], dropout=dropout)

        # self.final_proj = nn.Linear(self.config['descriptor_dim'], self.config['output_dim'])

        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

        # self.layer_norm = nn.LayerNorm(self.config['descriptor_dim'], eps=1e-6)

        if self.config.get('last_activation', False):
            if self.config['last_activation'].lower() == 'relu':
                self.last_activation = nn.ReLU()
            elif self.config['last_activation'].lower() == 'sigmoid':
                self.last_activation = nn.Sigmoid()
            elif self.config['last_activation'].lower() == 'tanh':
                self.last_activation = nn.Tanh()
            else:
                raise Exception('Not supported activation "%s".' % self.config['last_activation'])
        else:
            self.last_activation = None

    def forward(self, desc, kpts, normals):
        # import pdb;pdb.set_trace()
        ## Self boosting
        # Descriptor MLP encoder
        if self.denc is not None:
            desc = self.denc(desc)
        # Geometric MLP encoder
        if self.use_kenc:
            desc = desc + self.kenc(kpts)
            if self.use_dropout:
                desc = self.dropout(desc)

        # 法向量特征 encoder
        if self.use_normal:
            desc = desc + self.nenc(normals)
            if self.use_dropout:
                desc = self.dropout(desc)
        
        ## Cross boosting
        # Multi-layer Transformer network.
        if self.use_cross:
            # desc = self.attn_proj(self.layer_norm(desc))
            desc = self.attn_proj(desc)

        ## Post processing
        # Final MLP projection
        # desc = self.final_proj(desc)
        if self.last_activation is not None:
            desc = self.last_activation(desc)
        # L2 normalization
        if self.config['l2_normalization']:
            desc = F.normalize(desc, dim=-1)

        return desc

if __name__ == "__main__":
    from config import t1_featureboost_config
    fb_net = FeatureBooster(t1_featureboost_config)

    descs=torch.randn([1900,64])
    kpts=torch.randn([1900,65])
    normals=torch.randn([1900,3])

    import pdb;pdb.set_trace()

    descs_refine=fb_net(descs,kpts,normals)

    print(descs_refine.shape)
