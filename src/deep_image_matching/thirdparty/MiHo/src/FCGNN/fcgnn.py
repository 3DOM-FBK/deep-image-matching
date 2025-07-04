import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from pathlib import Path

class GNN(nn.Module):
    def __init__(self, depth=9):
        super().__init__()

        in_dim, r, self.n = 256, 20, 8
        AttnModule = Attention(in_dim=in_dim, num_heads=8)
        self.layers = nn.ModuleList([copy.deepcopy(AttnModule) for _ in range(depth)])

        self.embd_p = nn.Sequential(BasicBlock(self.n*4, in_dim, nn.Tanh()))
        self.embd_f = nn.Sequential(BasicBlock((2*r+1)**2*2, 3*in_dim), nn.LayerNorm(3*in_dim), BasicBlock(3*in_dim, in_dim))

        self.extract = ExtractPatch(r)

        self.mlp_s = nn.Sequential(OutBlock(in_dim, 1), nn.Sigmoid())
        self.mlp_o = nn.Sequential(OutBlock(in_dim, 2))

        local_path = Path(__file__).parent / 'weights/fcgnn.model'

        if os.path.exists(local_path):
            self.load_state_dict(torch.load(local_path, map_location='cpu')) 
        else:
            url = "https://github.com/xuy123456/fcgnn/releases/download/v0/fcgnn.model"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

        # print('FC-GNN weights loaded')
        
    def forward(self, img1, img2, matches):

        def _map(idx, n):

            id1 = idx.unsqueeze(1)
            id2 = idx.unsqueeze(2)
            
            relative_pos = id1 - id2
            b, l, c = idx.shape

            map = - (relative_pos**2).sum(dim=-1)

            n_neighbors = min(n, map.shape[1])
            
            b_ids = torch.arange(0, b, device=id1.device).view(-1, 1).repeat(1, l*n_neighbors).view(-1)
            l_ids = torch.arange(0, l, device=id1.device).view(-1, 1).repeat(1, b*n_neighbors).view(-1)
            
            _, query = map.topk(k=n_neighbors, dim=2, largest=True, sorted=True)

            relative_pos = relative_pos[b_ids, l_ids, query.view(-1), :].reshape(b, l, n_neighbors*4)

            if n_neighbors < n:
                relative_pos = torch.cat([relative_pos, torch.zeros((b, l, (n - n_neighbors)*4), device=relative_pos.device)], dim=2)

            return query, relative_pos

        f = self.extract(img1, img2, matches.long())
        f = self.embd_f(f)

        query, pos = _map(matches, self.n)

        x = self.embd_p(pos)
        x = x + f

        for layer in self.layers:
            x = layer(x, query)

        scores = self.mlp_s(x)
        offset = self.mlp_o(x)

        return offset, scores.squeeze(2)
    
    def optimize_matches(self, img1, img2, matches, thd=0.999, min_matches=10):

        if len(matches.shape) == 2:
            matches = matches.unsqueeze(0)

        matches = matches.round()
        offsets, scores = self.forward(img1, img2, matches)
        matches[:, :, 2:] = matches[:, :, 2:] + offsets[:, :, [1, 0]]
        mask = scores > thd

        new_matches = []
        for i in range(matches.shape[0]):

            if mask[i].sum() < min_matches:
                _, mask_i = scores[i].topk(k=min(matches.shape[1], min_matches))
            else:
                mask_i = mask[i]
        
            new_matches.append(matches[i][mask_i])
        
        return new_matches

    def optimize_matches_custom(self, img1, img2, matches, thd=0.999, min_matches=10):

        if len(matches.shape) == 2:
            matches = matches.unsqueeze(0)

        matches = matches.round()
        offsets, scores = self.forward(img1, img2, matches)
        matches[:, :, 2:] = matches[:, :, 2:] + offsets[:, :, [1, 0]]
        mask = scores[0] > thd
        
        if mask.sum() < min_matches:
            mask_i = scores[0].topk(k=min(matches.shape[1], min_matches))
            mask[mask_i[1]] = True
                
        return matches[0].detach(), mask

class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            act,
            nn.Linear(out_dim, out_dim, bias=False)
        )
    
    def forward(self, x):

        return self.layers(x)

class OutBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            act,
            nn.Linear(in_dim, out_dim, bias=False)
        )
    
    def forward(self, x):

        return self.layers(x)

class ExtractPatch(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, image1, image2, matches):

        def _padding(image, r):
            return torch.nn.functional.pad(image, (r, r, r, r))
        
        for i in range(matches.shape[0]):
            img1, img2 = image1[i:i+1], image2[i:i+1]
            
            match = matches[i]

            img1, img2 = _padding(img1, self.r), _padding(img2, self.r)

            d = 2*self.r+1

            lines = torch.arange(0, d, device=img1.device).view(-1, 1).float().repeat(1, d)
            columns = torch.arange(0, d, device=img1.device).view(1, -1).float().repeat(d, 1)
            M = torch.stack([lines, columns], dim=0).view(2, -1)
            M = torch.cat([M, M], dim=0).unsqueeze(0)

            match = match.unsqueeze(2)
            
            match = (match + M).long()

            p1 = img1[0, 0, match[:, 1, :].reshape(-1), match[:, 0, :].reshape(-1)].reshape(-1, (2*self.r+1)**2)
            p2 = img2[0, 0, match[:, 3, :].reshape(-1), match[:, 2, :].reshape(-1)].reshape(-1, (2*self.r+1)**2)

            p1_mean, p2_mean = p1.mean(dim=-1), p2.mean(dim=-1)
            p1_std,  p2_std  = p1.std(dim=-1),  p2.std(dim=-1)

            p1 = (p1 - p1_mean.unsqueeze(1)) / (p1_std.unsqueeze(1) + 1e-4)
            p2 = (p2 - p2_mean.unsqueeze(1)) / (p2_std.unsqueeze(1) + 1e-4)

            f1 = torch.cat([p1, p2], dim=1).unsqueeze(0)
            
            if i == 0:
                f = f1
            else:
                f = torch.cat([f, f1], dim=0)

        return f


class Attention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads

        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, in_dim, bias=False)

        self.merge = nn.Linear(in_dim, in_dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim, bias=False)
        )

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x0, query):
        
        x0_ = self.norm1(x0)

        q, k, v = self.q(x0_), self.k(x0_), self.v(x0_)

        B, qlen, n = query.shape
        mask = torch.zeros(B*qlen*n, device=x0.device)
        for i in range(B):
            mask[i*qlen*n:(i+1)*qlen*n] = i
        mask = mask.long()

        B, L, C= q.shape
        k = k[mask, query.view(-1), :].reshape(B, -1, n, C)
        v = v[mask, query.view(-1), :].reshape(B, -1, n, C)

        q = q.reshape(B, -1, 1, self.num_heads, C//self.num_heads)
        k = k.reshape(B, -1, n, self.num_heads, C//self.num_heads)
        v = v.reshape(B, -1, n, self.num_heads, C//self.num_heads)

        qk = (q * k).sum(dim=-1)
        sqrt_dk = 1. / q.shape[4]**.5
        A = F.softmax(sqrt_dk * qk, dim=2)
        q_values = (A.unsqueeze(4) * v).sum(dim=2).reshape(B, -1, C)

        q_values = self.merge(q_values)

        message = x0 + q_values
        x0 = x0 + self.norm2(self.mlp(message))

        return x0
