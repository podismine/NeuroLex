import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.nn import TransformerEncoderLayer
import math

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for MLP layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.mlp(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.norm(x + self.block(x))
    
class BNTF(nn.Module):
    def __init__(self, feature_dim, depth, heads, dim_feedforward, condition_dim=3,dim_reduction=4,input_num = 20):
        super().__init__()
        self.num_patches = input_num
        self.node_num = input_num
        self.dim_reduction = dim_reduction
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.node_num, self.node_num))
        self.condition_fusion = nn.Sequential(
            nn.Linear(self.node_num + 0, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            ResidualBlock(32),
            nn.Linear(32, dim_feedforward),# 16
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward,nhead=heads,dim_feedforward=dim_feedforward*2,dropout=0.1,batch_first=True,activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth,norm=torch.nn.LayerNorm(dim_feedforward))
        
        self.dim_reduction_layer = nn.Sequential(
            nn.Linear(dim_feedforward, self.dim_reduction),
            nn.GELU()
        )

        final_dim = self.dim_reduction * self.node_num
        self.g = MLPHead(final_dim, feature_dim * 2, feature_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        bz, _, _ = img.shape
        
        img = img + self.pos_embedding.expand(bz, -1, -1)
        
        x = self.condition_fusion(img)  # [batch_size, 10, node_num]
        x = self.encoder(x)
        
        x = self.dim_reduction_layer(x)
        x = x.reshape((bz, -1))
        x = self.g(x)
        
        return x
    
from einops import rearrange, repeat

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, distance='cos', 
                 anchor='closest', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embeddings
        self.embed_dim = embedding_dim
        self.beta = commitment_cost
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.max_codebook_misses_before_expiry = 50
        self.kmeans_iters = 10
        self.eps = 1e-5

        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', torch.zeros(num_embeddings, embedding_dim))

        codebook_misses = torch.zeros(num_embeddings)
        self.register_buffer('codebook_misses', codebook_misses)

    def init_embed_(self, data):
        embed = kmeans(data, self.num_embed, self.kmeans_iters)
        with torch.no_grad():
            self.embedding.weight.copy_(embed)
            self.embed_avg.data.copy_(embed.clone())
            self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, z):
        z_flattened = z.contiguous().view(-1, self.embed_dim)

        if not self.initted.item():
            self.init_embed_(z_flattened)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        encoding_indices = d.argmin(dim=1)
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        z_q = z + (z_q - z).detach()
        loss = self.beta * torch.mean((z_q.detach()-z)**2)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)) / avg_probs.shape[0]

        if self.training is True:
            ema_inplace(self.cluster_size, encodings.sum(0), self.decay)
            embed_sum = z_flattened.t() @ encodings
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.num_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / (cluster_size.unsqueeze(1) + self.eps)
            with torch.no_grad():
                self.embedding.weight.copy_(embed_normalized)
            self.expire_codes_(encoding_indices, z)
            
        return z_q, loss, encoding_indices, perplexity

    def expire_codes_(self, embed_ind, batch_samples):
        if self.max_codebook_misses_before_expiry == 0:
            return

        embed_ind = rearrange(embed_ind, '... -> (...)')
        used = torch.bincount(embed_ind, minlength=self.num_embed) > 0

        self.codebook_misses[used] = 0

        self.codebook_misses[~used] += 1

        expired_codes = self.codebook_misses >= self.max_codebook_misses_before_expiry
        if not torch.any(expired_codes):
            return

        self.codebook_misses[expired_codes] = 0
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask = expired_codes)

    def replace(self, samples, mask):
        num_expired = mask.sum().item()
        if num_expired == 0:
            return

        new_vecs = sample_vectors(samples, num_expired)  # shape: [num_expired, embed_dim]
        
        with torch.no_grad():
            self.embedding.weight[mask] = new_vecs.to(self.embedding.weight.dtype)

class Decoder(nn.Module):
    def __init__(self, d_model = 256, hidden_dim = 64, output_dim = 10, dim_reduction=4, heads = 2, depth = 2):
        super().__init__()
        self.node_num = output_dim
        self.dim_reduction = dim_reduction
        self.decoder_proj = MLPHead(d_model, self.dim_reduction * self.node_num * 2, self.dim_reduction * self.node_num)

        self.decoder_fusion = nn.Sequential(
            nn.Linear(self.dim_reduction + 0, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            ResidualBlock(32),
            nn.Linear(32, hidden_dim),# 16
        )
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=heads,dim_feedforward=hidden_dim*2,dropout=0.1,batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth,norm=torch.nn.LayerNorm(hidden_dim))
        self.g = MLPHead(hidden_dim, output_dim * 2, output_dim)
        
    def forward(self, x):
        N, D = x.shape
        proj_x = self.decoder_proj(x)
        proj_x = proj_x.view(N, self.node_num, self.dim_reduction)
        x = self.decoder_fusion(proj_x) # N, node_num, hidden_dim
        x = self.decoder(x) # N, node_num, hidden_dim
        reconstructed = self.g(x)
        reconstructed = (reconstructed + reconstructed.transpose(-1, -2)) / 2
        return reconstructed

class NeuroLex_model(nn.Module):
    def __init__(self, input_dim=7, condition_dim=1, latent_dim=16, hidden_dim=32, num_embeddings=32, commitment_cost=0.5, depth=4):
        super().__init__()
        self.node_num = input_dim

        self.encoder = BNTF(feature_dim = latent_dim, depth=depth, heads=4, dim_feedforward=hidden_dim, condition_dim=3,dim_reduction=4, input_num = input_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim=latent_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(d_model = latent_dim, hidden_dim = hidden_dim, output_dim = input_dim, dim_reduction=4, heads = 4, depth = depth)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # 稍小的gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param, gain=0.8)

    def forward(self, x: torch.Tensor):
        N, M, _ = x.shape
        
        encoded = self.encoder(x)
        
        quantized, vq_loss, token_idx, ppl = self.vector_quantizer(encoded)
        
        x_hat = self.decoder(quantized)
        
        return x_hat, vq_loss, ppl, token_idx.view(N, -1), encoded.view(N, -1)

    def infer_token(self, x: torch.Tensor):
        N, M, _ = x.shape
        encoded = self.encoder(x)
        quantized, vq_loss, token_idx, ppl = self.vector_quantizer(encoded)
        return token_idx.view(N, -1)

    def get(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, token_idx, ppl = self.vector_quantizer(encoded)
        x_hat = self.decoder(quantized)
        return encoded, quantized, token_idx, self.vector_quantizer.embedding.weight
