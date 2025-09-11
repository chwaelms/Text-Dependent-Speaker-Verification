#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Multi-Head Attention layer definition."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor,
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor = torch.empty(0),) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        
        # linear transformation for positional encoding
        """ 
        n_feat 차원의 입력을 받아서 동일한 n_feat 차원으로 변환하는 선형 변환
        가중치 행렬 W 크기 (n_feat, n_feat) <- y = xW^T
        위치 정보를 동일한 차원 공간 내에서 변환하면서도 원래의 차원을 유지하고자 할 때 사용"""
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        """ 
        pos_bias_u, v : 학습 가능한 파라미터 생성 (shape = [h, d_k])
        Xavier 초기화 방식으로 초기화 (신경망의 가중치를 초기화하는 방법, 각 뉴런의 출력이 적절한 범위를 가지도록 가중치 초기화)
        => 두 개의 바이어스를 사용하여 1. content score(입력 토큰들 간의 관계 모델링) 2. position score(상대적 위치 관계를 모델링)
        """
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor. 출력 : 변환된 텐서, 상대적 위치 인코딩이 반영
        """
        """ 
        Padding 추가 : 마지막 차원에 0 패딩을 추가하여 상대적 위치 계산에서 필요한 크기 보장
        차원 변환 및 변형 : 패딩된 텐서를 변형하여 상대적 위치 정보 계산 가능
        하삼각 행렬로 마스킹 (선택적) : 상삼각 행렬 요소를 0으로 설정하여 하삼각 행렬만 유지
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value) # attention의 입력으로 사용
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0) # positional embedding 변환 : 상대적 위치 임베딩을 선형 변환하여 self-attention에 적합한 형식으로 변환
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        ## bias 추가 및 attention scroe 계산
        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2) # Q에 learnable bias 추가
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1)) # Q와 K 간의 Attention score

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1)) # Q와 Positioonal Embedding 간의 Attention score
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)
        
        # 정규화
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask) # (batch, time, d_model)

class MultiViewRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention with Multi-View Self-Attention mechanism."""
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__(n_head, n_feat, dropout_rate)
        
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        # these two learnable bias are used in matrix c and matrix d
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        
    def generate_multi_view_mask(self,seq_len, layer_idx, device):
        """Generate head-specific masking matrix for multi-view self-attention based on layer index"""
        # 모든 값이 0인 3차원 텐서 생성
        mask = torch.zeros(self.h, seq_len, seq_len, device=device)
        for head in range(self.h):
            window_size = 2**layer_idx + 1 if head >=1 else 1
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[head, i, start:end] = 1
        return mask.bool()

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value) # attention의 입력으로 사용
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0) # positional embedding 변환 : 상대적 위치 임베딩을 선형 변환하여 self-attention에 적합한 형식으로 변환
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        ## bias 추가 및 attention scroe 계산
        ### Q에 learnable bias 추가
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2) # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2) # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # Q와 K 간의 Attention score
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        # Q와 Positioonal Embedding 간의 Attention score
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1)) # (batch, head, time1, time2)
        
        # 정규화
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)
        
        """Apply multi-view masking"""
        layer_idx = 0
        multi_view_mask = self.generate_multi_view_mask(scores.size(-1), layer_idx, device=scores.device)
        for head in range(self.h):
            scores[:, head] = scores[:, head].masked_fill(~multi_view_mask[head], float('-inf'))

        return self.forward_attention(v, scores, mask) # (batch, time, d_model)