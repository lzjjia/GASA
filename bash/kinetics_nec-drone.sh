# # ! kinetics-necdrone


#* adp unsup
python3 ../main.py \
    --train_source_dataset /data/liuxi/Dataset/k400-nec/Frame \
    --train_target_dataset /data/liuxi/Dataset/k400-nec/cropped/train_balanced \
    --val_dataset /data/liuxi/Dataset/k400-nec/cropped/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.0005\
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
    --name k-nec-biema \
    --project GASA-2025 \
    --da \
    --wandb \
    --use_queue \
    --queue_size 2048 \
    --pseudo_labels \
    --align_loss_weight 0.005 \
    --pretrained_source_model /data/liuxi/Project/GASA/bash/kinetics-nec-source-only=19.ckpt

#   \ --plot_feature_visualization \ --save_model \
