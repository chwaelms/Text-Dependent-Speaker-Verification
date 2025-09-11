import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal_Average_Pooling(nn.Module):
    def __init__(self, **kwargs):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(Temporal_Average_Pooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = torch.mean(x, axis=2)
        return x


class Temporal_Statistics_Pooling(nn.Module):
    def __init__(self, **kwargs):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(Temporal_Statistics_Pooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, axis=2)
        var = torch.var(x, axis=2)
        x = torch.cat((mean, var), axis=1)
        return x


class Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim):
        """SAP
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link： https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        Args:
            dim (pair): the size of attention weights 
        """
        super(Self_Attentive_Pooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim, 1))

    def forward(self, x):
        """Computes Self-Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        x = x.permute(0, 2, 1) 
        h = torch.tanh(self.sap_linear(x)) 
        w = torch.matmul(h, self.attention).squeeze(dim=2) 
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) 
        x = torch.sum(x * w, dim=1)
        return x


class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim):
        """ASP
        Paper: Attentive Statistics Pooling for Deep Speaker Embedding
        Link: https://arxiv.org/pdf/1803.10963.pdf
        Args:
            dim (pair): the size of attention weights (입력 텐서의 차원 크기/특징 크기)
        """
        super(Attentive_Statistics_Pooling, self).__init__()
        # 입력 데이터를 학습 가능한 선형 변환을 통해 변환 (dim->dim)
        self.sap_linear = nn.Linear(dim, dim)
        # 학습 가능한 attention 가중치 텐서 (dim x 1) -> 각 입력 프레임의 중요도 계산 시 사용
        self.attention = nn.Parameter(torch.FloatTensor(dim, 1))

    def forward(self, x):
        """Computes Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        x = x.permute(0, 2, 1) # (batch, dim, frames) -> (batch, frames, dim)
        h = torch.tanh(self.sap_linear(x)) # 선형 변환 적용 (dim -> dim) + 비선형 활성화 함수로 변환된 값 제한
        # 어텐션 가중치 계산
        w = torch.matmul(h, self.attention).squeeze(dim=2) # (batch, frames, 1) -> (batch, frames)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) # softmax 적용하여 각 "프레임"의 중요도를 확률 분포로 변환 : (batch, frames, 1)
        mu = torch.sum(x * w, dim=1) # (batch, dim) : 가중 평균 계산 = attention 가중치(w) x 입력 텐서(x)
        rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) ) # 가중 표준편차
        x = torch.cat((mu, rh), 1) # (batch, dim*2) 평균과 표준편차 연결
        return x # 입력 (batch, dim, frames) -> 출력 (batch, dim*2)

if __name__ == "__main__":
    data = torch.randn(10, 128, 100)
    pooling = Self_Attentive_Pooling(128)
    out = pooling(data)
    print(data.shape)
    print(out.shape)
