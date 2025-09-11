import torch
from wenet.transformer.encoder import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

class Conformer(torch.nn.Module):
    # def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
    #         pos_enc_layer_type="rel_pos"):

    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="multi_view_rel_pos"):

        super(Conformer, self).__init__()
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        self.pooling = AttentiveStatisticsPooling(output_size)
        self.bn = BatchNorm1d(input_size=output_size*2)
        self.fc = torch.nn.Linear(output_size*2, embedding_dim)
    
    def forward(self, feat):
        feat = feat.squeeze(1).permute(0, 2, 1) # (B, 1, n_mels, T) -> (B, n_mels, T) 변환
        lens = torch.ones(feat.shape[0]).to(feat.device) # 입력 데이터의 시퀀스 길이 계산
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens) # conformer 인코더를 통해 입력 데이터에서 특징 추출
        x = x.permute(0, 2, 1) # 데이터 차원 교환을 통해 풀링 레이어와 정규화 레이어 입력에 맞춤
        x = self.pooling(x)
        x = self.bn(x) # 정규화 적용
        x = x.permute(0, 2, 1)
        x = self.fc(x) # 최종 임베딩 생성
        x = x.squeeze(1) # 불필요한 차원 제거 (B, embedding_dim) 출력
        return x

# def conformer(n_mels=80, num_blocks=6, output_size=256, 
#         embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
#     model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
#             embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
#     return model

def conformer(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="multi_view_rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model



if __name__ == "__main__":
    for i in range(6, 7):
        print("num_blocks is {}".format(i))
        model = conformer(num_blocks=i)

        import time
        model = model.eval()
        time1 = time.time() # 실행 시간 측정
        with torch.no_grad(): # 그래디언트 계산 비활성화 (속도 향상)
            for i in range(100):
                data = torch.randn(1, 1, 80, 500) # 임의 데이터 생성
                embedding = model(data)  # 입력 데이터를 통해 임베딩 생성
        time2 = time.time()
        val = (time2 - time1)/100
        rtf = val / 5 # 모델이 실시간으로 데이터를 처리하는 성능 측정 지표

        total = sum([param.nelement() for param in model.parameters()]) # 모델 파라미터 수
        print("total param: {:.2f}M".format(total/1e6))
        print("RTF {:.4f}".format(rtf))
 
