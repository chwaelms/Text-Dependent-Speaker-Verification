import numpy as np

# ==============================
# Original cosine_score function
# ==============================
def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        labels.append(int(item[0]))
        scores.append(score)
    return labels, scores

# def cosine_score(trials, index_mapping, eval_vectors):
#     labels = []
#     scores = []
    
#     # 채널 쌍을 저장할 딕셔너리 생성
#     speaker_embeddings = {}
    
#     # 첫 번째 단계: 모든 임베딩을 화자와 세션별로 그룹화
#     for item in trials:
#         for path_idx in [1, 2]:  # enrollment와 test 경로 모두 처리
#             path = item[path_idx]
#             if path in index_mapping:
#                 filename = path.split('/')[-1]
#                 parts = filename.split('_')
                
#                 # 화자 ID와 세션 정보 추출 (SV0309_2_00_S0092.wav -> speaker=SV0309, session=2)
#                 speaker_id = parts[0]
#                 session = parts[1]
#                 channel = parts[2]  # 채널 정보
                
#                 # 키 생성: "화자ID_세션"
#                 speaker_session_key = f"{speaker_id}_{session}"
                
#                 if speaker_session_key not in speaker_embeddings:
#                     speaker_embeddings[speaker_session_key] = {}
                
#                 # 채널별 임베딩 저장
#                 speaker_embeddings[speaker_session_key][channel] = eval_vectors[index_mapping[path]]
    
#     # 두 번째 단계: trial 항목에 대해 평균화된 임베딩으로 점수 계산
#     for item in trials:
#         enroll_path = item[1]
#         test_path = item[2]
        
#         # enrollment와 test 파일의 화자 및 세션 정보 추출
#         enroll_filename = enroll_path.split('/')[-1]
#         test_filename = test_path.split('/')[-1]
        
#         enroll_parts = enroll_filename.split('_')
#         test_parts = test_filename.split('_')
        
#         enroll_speaker_session = f"{enroll_parts[0]}_{enroll_parts[1]}"
#         test_speaker_session = f"{test_parts[0]}_{test_parts[1]}"
        
#         # === ENROLLMENT 임베딩 가져오기 (평균화 적용) ===
#         if enroll_speaker_session in speaker_embeddings:
#             enroll_channels = speaker_embeddings[enroll_speaker_session]
            
#             # 채널이 하나 이상 있는지 확인
#             if len(enroll_channels) >= 1:
#                 # 모든 채널 임베딩의 평균 계산
#                 enroll_vector = np.mean(list(enroll_channels.values()), axis=0)
#             else:
#                 # 예외 상황 처리 (발생하지 않아야 함)
#                 if enroll_path in index_mapping:
#                     enroll_vector = eval_vectors[index_mapping[enroll_path]]
#                 else:
#                     continue  # 인덱스에 없으면 건너뛰기
#         else:
#             # 화자 세션 정보가 없는 경우 기본 enrollment 벡터 사용
#             if enroll_path in index_mapping:
#                 enroll_vector = eval_vectors[index_mapping[enroll_path]]
#             else:
#                 continue  # 인덱스에 없으면 건너뛰기
            
#         # === TEST 임베딩 가져오기 (평균화 적용) ===
#         if test_speaker_session in speaker_embeddings:
#             test_channels = speaker_embeddings[test_speaker_session]
            
#             # 채널이 하나 이상 있는지 확인
#             if len(test_channels) >= 1:  # 수정: 채널 수 늘어나면 이 부분을 >=1로 변경
#                 # 모든 채널 임베딩의 평균 계산
#                 test_vector = np.mean(list(test_channels.values()), axis=0)
#             else:
#                 # 하나의 채널만 있는 경우
#                 if test_path in index_mapping:
#                     test_vector = eval_vectors[index_mapping[test_path]]
#                 else:
#                     continue  # 인덱스에 없으면 건너뛰기
#         else:
#             # 화자 세션 정보가 없는 경우 기본 테스트 벡터 사용
#             if test_path in index_mapping:
#                 test_vector = eval_vectors[index_mapping[test_path]]
#             else:
#                 continue  # 인덱스에 없으면 건너뛰기
                
#         # 코사인 유사도 계산
#         score = enroll_vector.dot(test_vector.T)
#         denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
#         score = score/denom
        
#         labels.append(int(item[0]))
#         scores.append(score)
    
#     return np.array(labels), np.array(scores)