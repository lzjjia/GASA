import argparse
import os
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch import nn

from src.dataloader import prepare_data
from src.stam_utils import create_pretrained_model

# from src.utils import ConfusionMatrix
from src.utils import EpochCheckpointer, ConfusionMatrix
from src.model import TransformerVideoModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    SUP_OPT = ["sgd", "adam"]
    SUP_SCHED = ["reduce", "cosine", "step", "exponential", "none"]
    SUP_TRAINING = ["head", "head+partial", "head+temporal", "head+temporal-partial", "all"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_source_dataset", type=str, default='/data/liuxi/Dataset/k400-nec/Frame')
    parser.add_argument("--val_dataset", type=str, default='/data/liuxi/Dataset/k400-nec/cropped/test')
    parser.add_argument("--train_target_dataset", type=str, default='/data/liuxi/Dataset/k400-nec/cropped/train_balanced')

    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUP_OPT)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-9)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # scheduler
    parser.add_argument("--scheduler", choices=SUP_SCHED, default="cosine")
    parser.add_argument("--lr_steps", type=int, nargs="+")

    # general settings
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--train", type=str, choices=SUP_TRAINING, default='head+temporal')
    parser.add_argument("--replace_with_mlp", action="store_true", default=True)

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=16)

    # da
    parser.add_argument("--da", action="store_true", default=True)
    parser.add_argument("--source_only", action="store_true")
    parser.add_argument("--pseudo_labels", action="store_true", default=True)
    parser.add_argument("--transfer_loss_weight", type=float, default=1)
    parser.add_argument("--target_ce_loss_weight", type=float, default=0.0)

    parser.add_argument("--use_queue", action="store_true", default=True)
    parser.add_argument("--queue_size", type=int, default=2048)
    parser.add_argument("--align_loss_weight", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.2)

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16, choices=[8, 16, 32, 64])
    parser.add_argument("--n_clips", type=int, default=1)

    parser.add_argument("--pretrained_source_model", type=str, default='/data/liuxi/Project/GASA/bash/kinetics-nec-source-only=19.ckpt')

    # mlp stuff
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
    parser.add_argument("--mlp_n_layers", type=int, default=0)

    # wandb
    parser.add_argument("--name", default='k-nec-biema')
    parser.add_argument("--project", default='GASA-2025')
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--plot_feature_visualization", action="store_true")

    args = parser.parse_args()

    # find number of classes
    args.num_classes = len(set(os.listdir(args.train_source_dataset)))

    if args.source_only:
        assert args.da is None, "cannot do any adaptation with source only data"

    return args


def main():
    args = parse_args()
    # load backbone and weights
    backbone = create_pretrained_model(
        args.num_classes,
        path="/data/liuxi/Project/GASA/src/stam/stam_{}.pth".format(args.n_frames),
        n_frames=args.n_frames,
    )
    target_head = nn.Linear(768, args.num_classes)

    model = TransformerVideoModel(backbone, args.num_classes, target_head, args)
    if args.pretrained_source_model is not None:
        source_params = torch.load(args.pretrained_source_model, map_location="cuda")["state_dict"]
        source_params = {k.replace('transformer.', '', 1): v for k, v in source_params.items()}
        head_params = {}
        for k, v in source_params.items():
            if 'head' in k:
                head_params[k[16:]] = v
        model.transformer.load_state_dict(source_params, strict=False)
        model.target_head.load_state_dict(head_params)


    # dataloader
    train_loader, val_loader = prepare_data(
        args.train_source_dataset,
        args.val_dataset,
        train_target_dataset=args.train_target_dataset,
        n_frames=args.n_frames,
        n_clips=args.n_clips,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # add callbacks
    callbacks = []

    if args.save_model:
        checkpointer = EpochCheckpointer(args, frequency=6)
        callbacks.append(checkpointer)

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(name=args.name, project=args.project)
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger if args.wandb else None,
        distributed_backend="ddp",
        precision=args.precision,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    seed_everything(5)


    main()
