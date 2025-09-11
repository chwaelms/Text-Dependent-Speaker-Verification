#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Positionwise feed forward layer definition."""

import torch
import torch.nn as nn
from wenet.transformer.CTA import ChannelTemporalAttention


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim. 

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    ### 입력 크기와 출력 크기가 동일한 Position-wise Feed Forward Layer
    
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()): # Swish = nn.SiLU()
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        
        # idim(입력차원)에서 hidden_units(숨겨진 뉴런의 수)로 변환하는 첫번째 선형 변환
        """ Example
        (idim, hidden_units)=(4, 6)
            y = xW^T + b 연산 수행
            입력 벡터 x 크기 : 4
            가중치 행렬 크기 : 6x4
            편향 벡터 크기 : 6
            출력 벡터 크기 : 6
            => 4차원 입력이 6차원으로 변환 + 편향 6차원 = 출력 6차원
        """
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        
        # (수정 3) FFN + CFA
        # self.layer_norm = torch.nn.LayerNorm(hidden_units)
        # self.depthwise_1D = nn.Conv1d(
        #     in_channels=hidden_units,
        #     out_channels=hidden_units,  # in_channels와 동일
        #     kernel_size=3,  
        #     padding=1, 
        #     groups=hidden_units         # in_channels와 동일
        # )
        # self.attn = ChannelTemporalAttention(
        #     in_channels=hidden_units,
        #     kernel_size=3,
        #     middle_channels=8)
        # (수정 3) END.
        
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        # 두번째 선형 변환
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        
        

    ## 레이어의 순전파(Forward Pass) 정의
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D) = (batch size, sequence length, dimension) = (배치크기, 시퀀스 길이, 차원)
        Returns:
            output tensor, (B, L, D)
        """
        
        # (B, L, D) = (batch size, sequence length, dimension) = (배치크기, 시퀀스 길이, 차원)
        # forward pass 연산
        """ 연산 과정
        self.w_1(xs)
            첫 번째 선형 변환 수행 -> 입력 데이터를 숨겨진 뉴런 차원으로 변환
            결과 크기 (B, L, hidden_units)
        self.activation
        self.dropout
        self.w_2
            두 번째 선형 변환 수행 -> 숨겨진 차원을 다시 원래 입력 차원으로 되돌림
                결과 크기 (B, L, D) 형태의 텐서 반환
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
    
        # (수정 3) FFN + CFA
        # x = self.w_1(xs)
        # # print(f"x_w1: {x.shape}") # (1, 418, 2048) 
        # x = self.layer_norm(x)
        # # print(f"x_layer_norm: {x.shape}")
        # x = x.transpose(1, 2) # Depthwise conv을 위한 차원 변환 필요 (B, L, D) -> (B, D, L)
        # x = self.depthwise_1D(x)
        # # print(f"x_depthwise: {x.shape}")
        # x = x.transpose(1, 2)
        # x = self.activation(x)
        # # print(f"x_ReLU: {x.shape}")
        # w = self.attn(x)
        # # print(f"w_attn weight: {w.shape}")
        # x = x * w
        # # print(f"x_attn: {x.shape}")
        # x = self.dropout(x)
        # x = self.w_2(x)
        # # print(f"x_w2: {x.shape}")
        # return x
