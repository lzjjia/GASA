# # ! kinetics-necdrone

#* source only
#python3 ../main.py \
#    --train_source_dataset /data/liuxi/Dataset/k400-nec/Frame \
#    --val_dataset /data/liuxi/Dataset/k400-nec/cropped/test \
#    --epochs 20 \
#    --optimizer sgd \
#    --lr 0.001 \
#    --weight_decay 1e-9 \
#    --scheduler cosine \
#    --batch_size 8 \
#    --n_clips 1 \
#    --n_frames 16 \
#    --frame_size 224 \
#    --num_workers 5 \
#    --gpus 0  \
#    --train head+partial \
#    --mlp_hidden_dim 2048 \
#    --mlp_n_layers 0 \
#    --replace_with_mlp \
#    --name kinetics-nec-source-head+partial-bitrans \
#    --project transformer_da \
#    --save_model \
#    --wandb

#
#* adp unsup
python3 ../main.py \
    --train_source_dataset /data/liuxi/Dataset/k400-2-UAV_Human/croped/Frame_img \
    --train_target_dataset /data/liuxi/Dataset/k400-2-UAV_Human/croped/train_frame_crop \
    --val_dataset /data/liuxi/Dataset/k400-2-UAV_Human/croped/test_frame_crop \
    --epochs 15 \
    --optimizer sgd \
    --lr 0.005\
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 64 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 4 \
    --gpus 0  \
    --train head+temporal \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name kinetics-uav-0813 \
    --project bmvc-2021 \
    --da \
    --wandb \
    --use_queue \
    --queue_size 2048 \
    --pseudo_labels \
    --align_loss_weight 0.005 \
    --pretrained_source_model /data/liuxi/Project/UDAVT-main/bash/trained_models/bq3kl1mi/kinetics-uav-source-head+partial-ep=19.ckpt

#   \ --plot_feature_visualization \ --save_model \
##* adp sup
#python3 ../main.py \
#    --train_source_dataset /data/liuxi/Dataset/k400-nec/Frame \
#    --train_target_dataset /data/liuxi/Dataset/k400-nec/cropped/train_balanced \
#    --val_dataset /data/liuxi/Dataset/k400-nec/cropped/test \
#    --epochs 20 \
#    --optimizer sgd \
#    --lr 0.02 \
#    --weight_decay 1e-9 \
#    --scheduler cosine \
#    --batch_size 64 \
#    --n_clips 1 \
#    --n_frames 16 \
#    --frame_size 224 \
#    --num_workers 4 \
#    --gpus 0 \
#    --train head+temporal \
#    --mlp_hidden_dim 2048 \
#    --mlp_n_layers 0 \
#    --replace_with_mlp \
#    --name kinetics-nec-sup \
#    --project bmvc-2021 \
#    --da \
#    --use_queue \
#    --queue_size 2048 \
#    --align_loss_weight 0.005 \
#    --wandb \
#    --pretrained_source_model /data/liuxi/Project/UDAVT-main/bash/trained_models/bq3kl1mi/kinetics-nec-source-head+partial-bq3kl1mi-ep=19.ckpt

