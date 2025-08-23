import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

import random

# include in configs.py

# feature_dim = 512 # referred to as D in footnote
# num_heads = 8
# dropout = 0.2

class CrossModalGatedAttention(nn.Module):
  def __init__(self, feature_dim=512, num_heads=8, dropout=0.2):
    super(CrossModalGatedAttention, self).__init__()

    self.num_heads = num_heads
    self.head_dim = feature_dim // num_heads
    self.feature_dim = feature_dim

    self.scale = self.feature_dim ** -0.5
    self.Scale = self.head_dim ** -0.5

    assert feature_dim % num_heads == 0

    # cross-modality attention
    self.qkv_i = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.qkv_j = nn.Linear(feature_dim,feature_dim*3, bias = False)
 #   self.proj = nn.Linear(feature_dim, feature_dim) <- not used
    self.att_drop = nn.Dropout(dropout)
 #   self.proj_drop = nn.Dropout(dropout) <- not used
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

    # forget gate weights
    self.W_f = nn.Linear(feature_dim*2, feature_dim, bias=True)
    self.W_m = nn.Linear(feature_dim, feature_dim, bias=True)

    # fusion layer attention weights
    self.QKV = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.Proj = nn.Linear(feature_dim, feature_dim)
    self.Att_drop = nn.Dropout(dropout)
    self.Proj_drop = nn.Dropout(dropout)

    # initialize weights
    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.qkv_i.weight)
    nn.init.xavier_uniform_(self.qkv_j.weight)
    nn.init.xavier_uniform_(self.QKV.weight)
#    nn.init.xavier_uniform_(self.proj.weight) <- not used
    nn.init.xavier_uniform_(self.Proj.weight)
    nn.init.xavier_uniform_(self.W_f.weight)
    nn.init.xavier_uniform_(self.W_m.weight)
#    nn.init.constant_(self.proj.bias, 0) <- not used
    if self.Proj.bias is not None:
      nn.init.constant_(self.Proj.bias, 0)

  @staticmethod
  def _ensure_3d(x):
    if x.dim() == 2:
      return x.unsqueeze(1)
    elif x.dim() == 3:
      return x
    raise ValueError("input must be (B x D) or (B x N x D)")

  def forward(self, x_1, x_2):
    # each input: (B x N x D)
    # output:  (B x N x D)
    B_1, N_1, F_1 = x_1.shape
    B_2, N_2, F_2 = x_2.shape
    if B_1 != B_2:
      raise ValueError("batch size does not match for two modalities")
    if N_1 != N_2:
      raise ValueError("the number of frames does not match for two modalities")
    if F_1 != F_2:
      raise ValueError("the number of features does not match for two modalities")

    batch_size = B_1
    num_frames = N_1
    feature_dim = F_1

    # (1) cross-attention > forget gate
    z_i, z_j = self._ensure_3d(x_1), self._ensure_3d(x_2)

    qkv_i = self.qkv_i(z_i)# (B x N x 3D)
    qkv_j = self.qkv_j(z_j) # (B x N x 3D)
    qkv_i = qkv_i.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)
    qkv_j = qkv_j.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)

    _, k_i, v_i = qkv_i.permute(2, 0, 1, 3) # each: (B x N x D)
    q_j, _, _ = qkv_j.permute(2, 0, 1, 3) # each: (B x N x D)

      # (video, audio) pair # (audio, video)의 reverse order는 생략 bcs performance 차이 insignificant according to the paper
    a_ij = F.softmax((q_j @ k_i.transpose(-2, -1) * self.scale), dim=-1)  # (B x N x N)
    a_ij = self.att_drop(a_ij) @ v_i # (B x N x D)
    f_ij = self.sigmoid(self.W_f(torch.concat((a_ij, z_j), dim = 2))) # (B x N x D)
    h_ij = self.relu(z_i + (self.W_m(a_ij) *f_ij)) # (B x N x D)

    # (2) Fusion Layer
    QKV = self.QKV(h_ij) # (B x N x 3D)
    QKV = QKV.reshape(batch_size, num_frames, 3, self.num_heads, self.head_dim) # (B x N x 3 x Nh x Dh) : D -> Nh x Dh
    Q, K, V = QKV.permute(2, 0, 3, 1, 4) # each: (B x Nh x N x Dh)

    Attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.Scale, dim=-1) # (B x Nh x N x N)
    H = self.Att_drop(Attn) @ V # (B x Nh x N x Dh)
    H = H.transpose(1, 2).reshape(batch_size, num_frames, -1) # (B x N x D): Nh x Dh > D)
    H = self.Proj(H) # (B x N x D)
    H = self.Proj_drop(H) # (B x N x D)

    return H