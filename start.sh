# =====================================
#             1. 모델 구성
# =====================================
encoder_name="conformer_cat" # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192
loss_name="aamsoftmax" # softmax | amsoftmax | aamsoftmax

dataset="vox"
num_blocks=6 # 3 | 4 | 5 | 6 (conformer 블록 개수)

# =====================================
#   2. 데이터세트 및 trial path 설정
# =====================================
# (TI-SV) AISHELL dataset --------------------   
# num_classes=1991
# train_csv_path="data/AISHELL_train.csv"
# trial_path=data/AISHELL_test.txt

# (TD-SV) FFSVC2020 task1 + HI-MIA dataset --------------------   
num_classes=374
train_csv_path="data/train_TD.csv"
trial_path=data/SLR_8-8.txt

# (TD-SV) HI-MIA dataset --------------------
# num_classes=254
# train_csv_path="data/train_SLR.csv"
# trial_path=data/SLR_8-8.txt


# =====================================
#             3. 학습 설정
# =====================================
input_layer=conv2d2
pos_enc_layer_type=rel_pos # no_pos| rel_pos| multi_view_rel_pos
save_dir=experiment/${input_layer}/${encoder_name}_${num_blocks}_${embedding_dim}_${loss_name}_anything

# transfer learning (TF) 시 사용할 pre-trained 모델 경로 (TF이 아닐 경우 주석처리)
checkpoint_path=/home/nvidia/Desktop/conformer/epoch=8_cosine_eer=1.36.ckpt # pre-train TI-SV model.7z

mkdir -p $save_dir
cp start.sh $save_dir
cp main.py $save_dir
cp -r module $save_dir
cp -r wenet $save_dir
cp -r scripts $save_dir
cp -r loss $save_dir
echo save_dir: $save_dir

# =====================================
#  TI-SV 학습 시 사용한 하이퍼파라미터
# =====================================
# python3 main.py \
#         --second 2 \
#         --batch_size 100 \
#         --num_workers 20 \
#         --max_epochs 200   \
#         --embedding_dim $embedding_dim \
#         --save_dir $save_dir \
#         --encoder_name $encoder_name \
#         --train_csv_path $train_csv_path \
#         --learning_rate 0.001 \
#         --encoder_name ${encoder_name} \
#         --num_classes $num_classes \
#         --trial_path $trial_path \
#         --loss_name $loss_name \
#         --num_blocks $num_blocks \
#         --warmup_step 2000 \
#         --step_size 4 \
#         --gamma 0.5 \
#         --weight_decay 0.000001   \
#         --input_layer $input_layer \
#         --pos_enc_layer_type $pos_enc_layer_type


# =====================================
#  TD-SV 학습 시 사용한 하이퍼파라미터
# =====================================
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
        --second 2 \
        --batch_size 64 \
        --num_workers 20 \
        --max_epochs 30 \
        --embedding_dim $embedding_dim \
        --save_dir $save_dir \
        --encoder_name $encoder_name \
        --train_csv_path $train_csv_path \
        --learning_rate 0.0002 \
        --encoder_name ${encoder_name} \
        --num_classes $num_classes \
        --trial_path $trial_path \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --warmup_step 1000 \
        --step_size 2 \
        --gamma 0.5 \
        --weight_decay 0.00001   \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $checkpoint_path \
        --aug \
        --fine_tune \
        --causal False 