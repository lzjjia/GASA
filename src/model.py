import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import wandb
from einops import rearrange
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from loss import InterDomainBasedNCE, RelationNetwork
from . import CSA
from .attention_module import MultiheadAttention, MultiheadAttention2

try:
    from gather_layer import gather
    from modules import CDAN, AdversarialNetworkCDAN, RandomLayer
    from utils import accuracy_at_k, compute_acc, mmd2
except ImportError:
    from .gather_layer import gather
    from .modules import CDAN, AdversarialNetworkCDAN, RandomLayer
    from .utils import accuracy_at_k, compute_acc, mmd2


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class cross_attention_learning(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(cross_attention_learning, self).__init__()
        self.conv1 = ConvBlock(input_size, hidden_size, 1)  # kernelLearner
        self.conv2 = nn.Conv1d(hidden_size, input_size, 1)

    def forward(self, f_s, f_t):
        f_s = F.normalize(f_s, p=2, dim=-1, eps=1e-12)
        f_t = F.normalize(f_t, p=2, dim=-1, eps=1e-12)

        f_s = f_s.unsqueeze(2)
        f_t = f_t.unsqueeze(2)
        R = torch.matmul(f_s, f_t.transpose(1, 2))
        # R = torch.matmul(f_s.transpose(1, 2), f_s)
        # R2 = torch.matmul(f_t.transpose(1, 2), f_t)# B,768,768
        w = F.relu(self.conv1(R.mean(2).unsqueeze(2)))  # B,512,1
        w = self.conv2(w)  # B,768,1
        w = F.softmax(torch.matmul(R, w).squeeze(2), dim=-1)
        return w  # B,768


class Mlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        # projector
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.projector(x)

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
class TsallisEntropy(nn.Module):

    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape

        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)

        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def weighted_mean(outputs, key, batch_size_key):
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def ib_loss_func(z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3):
    N, D = z1.size()  # N:B, D:2048

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)
    # corr = cos_sim(z1, z2)
    corr = torch.einsum("bi, bj -> ij", z1, z2) / N  # matrix's lie x
    # Lib = &(1-Cii)^2+lamb&&(Cij)^2
    diag = torch.eye(D, device=corr.device)  # (2048,2048)
    cdif = (corr - diag).pow(2)  # (2048,2048)
    cdif[~diag.bool()] *= lamb
    loss = cdif.sum()
    return loss


def compute_ib_loss(
    self,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    source_queue: torch.Tensor = None,
    source_queue_y: torch.Tensor = None,
):

    z1 = []
    z2 = []

    for c in range(self.num_classes):
        source_indexes = (y_source == c).view(-1).nonzero()
        target_indexes = (y_target == c).view(-1).nonzero()
        for i, j in product(source_indexes, target_indexes):
            z1.append(z_s[i])
            z2.append(z_t[j])

    # handle queues
    if source_queue is not None:
        for c in range(self.num_classes):
            source_indexes = (source_queue_y == c).view(-1).nonzero()
            target_indexes = (y_target == c).view(-1).nonzero()
            for i, j in product(source_indexes, target_indexes):
                z1.append(source_queue[i])
                z2.append(z_t[j])

    n_pairs = len(z1)
    if n_pairs > 2:
        z1 = torch.cat(z1)
        z2 = torch.cat(z2)
        relation_pairs = torch.cat((z1, z2), 1)
        relations = self.relationNetwork(relation_pairs)
        relations = 1 - relations
        loss = ib_loss_func(z1, z2 * relations + z2)
        # loss = ib_loss_func(z1, z2)
    else:
        loss = torch.tensor(0.0, device=self.device)

    return loss, n_pairs


class ClassMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers):
        super().__init__()

        if n_layers:
            self.model = []
            self.model.append(nn.Linear(in_dim, hidden_dim))
            self.model.append(nn.BatchNorm1d(hidden_dim))
            self.model.append(nn.ReLU(hidden_dim))
            for _ in range(n_layers - 1):
                self.model.append(nn.Linear(hidden_dim, hidden_dim))
                self.model.append(nn.BatchNorm1d(hidden_dim))
                self.model.append(nn.ReLU(hidden_dim))
            self.model = nn.Sequential(*self.model)
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.model = nn.Identity()
            self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        feat = self.model(x)
        out = self.classifier(feat)
        return feat, out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        # projector
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.projector(x)


class TransformerVideoModel(pl.LightningModule):
    def __init__(
            self,
            transformer,
            num_classes,
            target_head,
            args,
    ):
        super().__init__()
        # define base model
        self.transformer = transformer
        self.num_classes = num_classes
        self.args = args

        self.target_head = target_head
        self.target_ema_updater = EMA(0.99)
        self.VCA = cross_attention_learning(2048, 400)
        self.inter_domain_based_contrastive = InterDomainBasedNCE(temperature=0.1)
        self.CSA_block_att = CSA.AttentionalGNN(768).to(self.device)
        self.relationNetwork = RelationNetwork(4096, 400)
        self.dec1 = nn.Sequential(
            nn.Linear(2048, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )
        self.recon_fn = torch.nn.L1Loss()
        self.dec2 = self.dec1  # share decoder
        # self.dec2 = nn.Sequential(
        #     nn.Linear(2048, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768),
        # )
        self.l2norm = lambda x: F.normalize(x, dim=-1)

        if self.args.use_queue:
            # queue
            self.queue_size = args.queue_size
            self.register_buffer("source_queue", torch.randn(self.queue_size, args.mlp_hidden_dim))
            self.register_buffer("target_queue", torch.randn(self.queue_size, args.mlp_hidden_dim))
            self.register_buffer("source_queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("target_queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # replace head with mlp
        if self.args.replace_with_mlp:
            self.transformer.head = ClassMLP(
                self.transformer.embed_dim, args.mlp_hidden_dim, num_classes, args.mlp_n_layers
            )

        if self.args.da:
            if args.mlp_n_layers:
                in_dim = args.mlp_hidden_dim
            else:
                in_dim = self.transformer.embed_dim

            self.target_encoder = ProjectionHead(in_dim, args.mlp_hidden_dim, args.mlp_hidden_dim)
            self.source_encoder = ProjectionHead(in_dim, args.mlp_hidden_dim, args.mlp_hidden_dim)


        self.set_training()

    @torch.no_grad()
    def dequeue_and_enqueue(self, z_s, z_t, y_s, y_t):
        z_s = gather(z_s)
        y_s = gather(y_s)
        z_t = gather(z_t)
        y_t = gather(y_t)

        batch_size = z_s.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.source_queue[ptr: ptr + batch_size, :] = z_s
        self.source_queue_y[ptr: ptr + batch_size] = y_s
        self.target_queue[ptr: ptr + batch_size, :] = z_t
        self.target_queue_y[ptr: ptr + batch_size] = y_t
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def configure_optimizers(self):
        args = self.args

        # select optimizer
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD
            extra_optimizer_args = {"momentum": 0.9}
        else:
            optimizer = torch.optim.Adam
            extra_optimizer_args = {}

        # filter parameters to train
        if args.train == "head":
            parameters = self.transformer.head.parameters()

        elif args.train == "head+partial":
            to_keep = ["norm", "head", "pos_embed", "cls_token", "patch_embed"]
            parameters = []
            for name, p in self.named_parameters():
                if any(keep_name in name for keep_name in to_keep):
                    parameters.append(p)

        elif args.train == "head+temporal":
            parameters = list(self.transformer.head.parameters()) + list(self.transformer.aggregate.parameters()) + \
                         list(self.CSA_block_att.parameters()) + list(self.source_encoder.parameters()) + \
                         list(self.VCA.parameters()) + list(self.dec1.parameters()) + list(self.relationNetwork.parameters()) + list(self.target_head.parameters())

        elif args.train == "head+temporal-partial":
            to_keep = ["norm", "pos_embed", "cls_token", "patch_embed"]
            parameters = list(self.transformer.head.parameters())
            for name, p in self.transformer.aggregate.named_parameters():
                if any(keep_name in name for keep_name in to_keep):
                    parameters.append(p)

        elif args.train == "all":
            parameters = self.transformer.parameters()

        else:
            raise ValueError(f"{args.train} not in (head, head+partial, everything)")

        optimizer = optimizer(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            **extra_optimizer_args,
        )

        # select scheduler
        if args.scheduler == "none":
            return optimizer
        else:
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs))
            elif args.scheduler == "reduce":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            elif args.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps)
            elif args.scheduler == "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.weight_decay)

            return [optimizer], [scheduler]

    def set_training(self):
        if self.args.train == "head":
            self.transformer.eval()
            self.transformer.head.train()
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.transformer.head.parameters():
                param.requires_grad = True

        elif self.args.train == "head+partial":
            to_keep = ["norm", "head", "pos_embed", "cls_token", "patch_embed"]
            for name, p in self.named_parameters():
                if not any(keep_name in name for keep_name in to_keep):
                    p.requires_grad = False

        elif self.args.train == "head+temporal":
            self.transformer.eval()
            self.transformer.aggregate.train()
            self.transformer.head.train()
            for param in self.transformer.parameters():
                param.requires_grad = False
            to_train = list(self.transformer.head.parameters()) + list(self.transformer.aggregate.parameters()) + \
                       list(self.CSA_block_att.parameters()) + list(self.relationNetwork.parameters()) + list(self.source_encoder.parameters())+\
                       list(self.VCA.parameters()) + list(self.dec1.parameters()) + list(self.target_head.parameters())
            for param in to_train:
                param.requires_grad = True

        elif self.args.train == "head+(temporal-partial)":
            self.transformer.eval()
            self.transformer.head.train()
            self.transformer.aggregate.train()

            for param in self.transformer.parameters():
                param.requires_grad = False

            for param in self.transformer.head.parameters():
                param.requires_grad = True
            to_keep = ["norm", "pos_embed", "cls_token", "patch_embed"]
            for name, p in self.transformer.aggregate.named_parameters():
                if not any(keep_name in name for keep_name in to_keep):
                    p.requires_grad = False

        if hasattr(self, "source_encoder"):
            for param in self.source_encoder.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        self.set_training()

    def forward(self, x):
        # x = rearrange(x, "b n_clips c f h w -> (b n_clips f) c h w")
        feat = self.transformer(x)
        return feat

    def forward_att(self, x):
        b = x.size(0)
        outs = []
        atts = []
        for clip in range(x.size(1)):
            partial = x[:, clip]
            partial = rearrange(partial, "b c f h w -> (b f) c h w")
            out, att = self.transformer.forward_att(partial)
            att = att[:, 0, 1:]
            out = rearrange(out, "(b n_clips) f -> b n_clips f", b=b).detach()
            att = rearrange(att, "(b n_clips) f -> b n_clips f", b=b).detach()

            outs.append(out)
            atts.append(att)

        out = torch.cat(outs, dim=1)
        att = torch.cat(atts, dim=1)
        return out, att

    def single_domain_training_step(self, X, y):
        args = self.args
        log = {}

        # apply model
        feat = self.transformer(X)
        out = self.head(feat)

        if args.pseudo_labels:
            # get target pseudo-label
            pseudo_y = out.detach().argmax(dim=1)

            if self.args.supervised_labels:
                loss = F.cross_entropy(out, y)
            else:
                loss = F.cross_entropy(out, pseudo_y)

            # compute pseudo-labels accuracies and number of pseudo-labels
            pseudo_labels_acc = compute_acc(y, pseudo_y, self.device)

            # update log
            log["pseudo_labels_acc"] = pseudo_labels_acc

        else:  # supervised
            loss = F.cross_entropy(out, y)

        acc1, acc5 = accuracy_at_k(out, y, top_k=(1, 5))
        log.update({"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5})

        return loss, log

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.target_encoder, self.source_encoder)

    def update_moving_average2(self):
        update_moving_average(self.target_ema_updater, self.dec2, self.dec1)

    def multi_domain_training_step(self, X_source, y_source, X_target, y_target, batch_idx):
        args = self.args

        log = {}
        # apply model   feat(B,768), out(B, 19)
        feat_s = self.transformer(X_source)
        feat_t = self.transformer(X_target)

        # As = self.VCA(feat_s, feat_t)
        # feat_s = feat_s * (As + 1)
        # feat_t = feat_t * (As + 1)

        feat_s, out_s = self.transformer.head(feat_s)
        out_t = self.target_head(feat_t)

        # with torch.no_grad():
        _, out_t_2 = self.transformer.head(feat_t)
        max_prob, pseudo_y = torch.max(F.softmax(out_t_2, dim=1), dim=-1)

        source_ce_loss = F.cross_entropy(out_s, y_source)
        loss = source_ce_loss  # source_ce_loss_weight default 1
        log["train_source_ce_loss"] = source_ce_loss

        ts_loss = TsallisEntropy(temperature=2.5, alpha=1.5)
        transfer_loss = ts_loss(out_t_2)
        loss += transfer_loss
        log["transfer_loss"] = transfer_loss

        if not args.source_only:
            # compute pseudo labels if needed
            if args.pseudo_labels:
                target_ce_loss = F.cross_entropy(out_t, pseudo_y)
                pseudo_labels_acc = compute_acc(y_target, pseudo_y, self.device)

                temp = F.softmax(out_t, dim=1).topk(2, dim=1)[0]
                diff_top2 = (temp[:, 0] - temp[:, 1]).mean()
                log.update(
                    {
                        "pseudo_labels_acc": pseudo_labels_acc,
                        "n_unique_pseudo_labels": pseudo_y.unique().size(0),
                        "pseudo_label_avg_prob_diff_between_1_and_2": diff_top2,
                    }
                )

            else:  # supervised
                pseudo_y = y_target
                # cross entropy on target
                target_ce_loss = F.cross_entropy(out_t, y_target)
            loss += target_ce_loss
            log["train_target_ce_loss"] = target_ce_loss

            # ****** DA part ******
            if self.args.da:
                z_s = self.source_encoder(feat_s)  # (B,2048)
                z_t = self.target_encoder(feat_t)  # (B,2048)

                # z_s, z_t = self.CSA_block_att(feat_s, feat_t, z_s, z_t)
                As = self.VCA(z_s, z_t)
                z_s = z_s * (As + 1)
                z_t = z_t * (As + 1)

                selection_source = torch.ones(y_source.size(0), dtype=bool, device="cuda")
                selection_target = (max_prob >= 0.99)  # pseudo is reliable
                selection = torch.cat((selection_source, selection_target), dim=0)
                IDC_loss = self.inter_domain_based_contrastive(z_s, z_t, y_source,
                                                               pseudo_y, selection=selection)
                loss += IDC_loss
                log["IDC_loss"] = IDC_loss

                # if self.args.use_queue:
                #     align_loss, n_pairs = compute_ib_loss(
                #         self,
                #         z_s,
                #         z_t,
                #         y_source,
                #         pseudo_y,
                #         self.source_queue,
                #         self.source_queue_y,
                #     )
                #
                # loss += self.args.align_loss_weight*align_loss
                # log["train_align_loss"] = self.args.align_loss_weight*align_loss
                # log["n_pairs"] = n_pairs

                fake_z_s = self.l2norm(self.dec1(z_s))  # (B,768)
                fake_z_t = self.l2norm(self.dec2(z_t))  # (B,768)
                loss_recon_s = self.recon_fn(fake_z_s, feat_s)
                loss_recon_t = self.recon_fn(fake_z_t, feat_t)

                loss += loss_recon_s
                log["loss_recon_s"] = loss_recon_s
                loss += loss_recon_t
                log["loss_recon_t"] = loss_recon_t

                # enqueue elements
                # if self.args.use_queue:
                #     self.dequeue_and_enqueue(z_s, z_t, y_source, pseudo_y)
        self.update_moving_average()
        # self.update_moving_average2()
        source_acc1, source_acc5 = accuracy_at_k(out_s, y_source, top_k=(1, 5))
        target_acc1, target_acc5 = accuracy_at_k(out_t, y_target, top_k=(1, 5))
        log.update(
            {
                "train_loss": loss,
                "train_source_acc1": source_acc1,
                "train_source_acc5": source_acc5,
                "train_target_acc1": target_acc1,
                "train_target_acc5": target_acc5,
            }
        )

        return loss, log, feat_s, feat_t

    def training_step(self, batch, batch_idx):
        # non-source free variant
        if len(batch) == 4:
            X_source, y_source, X_target, y_target = batch
            loss, log, feat_s, feat_t = self.multi_domain_training_step(
                X_source, y_source, X_target, y_target, batch_idx
            )
            ret = {
                "loss": loss,
                "y_source": y_source,
                "y_target": y_target,
                "feat_s": feat_s,
                "feat_t": feat_t,
            }
        # source free
        else:
            X, y = batch
            loss, log = self.single_domain_training_step(X, y)
            ret = {"loss": loss, "y": y}

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return ret

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        feat = self.transformer(X)
        out = self.target_head(feat)
        loss = F.cross_entropy(out, target).detach()

        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "outputs": out,
            "targets": target,
            "y_target": target,
            "feat_t": feat,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = weighted_mean(outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outputs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)
