import argparse

import cv2
import numpy as np
import torchvision.transforms.functional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from thop import profile
import torch

from src.stam_utils import create_pretrained_model
# from src.video_model_bitrain import TransformerVideoModel

from src.model import TransformerVideoModel

def parse_args():
    SUP_OPT = ["sgd", "adam"]
    SUP_SCHED = ["reduce", "cosine", "step", "exponential", "none"]
    SUP_TRAINING = ["head", "head+partial", "head+temporal", "head+temporal-partial", "all"]

    parser = argparse.ArgumentParser()
    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUP_OPT)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # scheduler
    parser.add_argument("--scheduler", choices=SUP_SCHED, default="reduce")
    parser.add_argument("--lr_steps", type=int, nargs="+")

    # general settings
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--train", type=str, default='head+temporal')
    parser.add_argument("--replace_with_mlp", default=True)

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--precision", type=int, default=16)

    # da
    parser.add_argument(
        "--da",
        type=str,
        default='ib',
        choices=["adversarial", "mmd2", "cdan", "ib", "vicreg", "simclr"],
    )
    parser.add_argument("--source_only", default=False)
    parser.add_argument("--pseudo_labels", action="store_true")
    parser.add_argument("--transfer_loss_weight", type=float, default=1)
    parser.add_argument("--target_ce_loss_weight", type=float, default=0.0)

    parser.add_argument("--use_queue", action="store_true")
    parser.add_argument("--queue_size", type=int, default=2048)

    # adversarial
    parser.add_argument("--adversarial_loss_weight", type=float, default=1.0)
    parser.add_argument("--adversarial_coeff", type=float, default=-1.0)
    parser.add_argument("--source_ce_loss_weight", type=float, default=1.0)

    # mmd
    parser.add_argument("--mmd_loss_weight", type=float, default=1.0)

    # ib
    parser.add_argument("--ib_loss_weight", type=float, default=1.0)
    parser.add_argument("--mse_loss_weight", type=float, default=0.25)
    # vicreg
    parser.add_argument("--vicreg_loss_weight", type=float, default=1.0)
    parser.add_argument("--sim_loss_weight", type=float, default=25.0)
    parser.add_argument("--var_loss_weight", type=float, default=25.0)
    parser.add_argument("--cov_loss_weight", type=float, default=1.0)

    # simclr
    parser.add_argument("--simclr_loss_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16, choices=[8, 16, 32, 64])
    parser.add_argument("--n_clips", type=int, default=1)

    parser.add_argument("--pretrained_source_model", type=str, default=None)

    # mlp stuff
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
    parser.add_argument("--mlp_n_layers", type=int, default=0)

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot_feature_visualization", action="store_true")

    args = parser.parse_args()

    return args
img = torch.randn(1,1, 3, 8, 224, 224).cuda()
label= torch.tensor([0]).cuda()
args = parse_args()
print("-"*20,"our","-"*20)
model1 = create_pretrained_model(
        7,
        n_frames=16,
    )
model2 = create_pretrained_model(
        7,
        n_frames=16,
    )
model = TransformerVideoModel(model1,model2,19,args)
model.cuda()
params = torch.load('/bash/trained_models/g7cs1lm9_uavcrop_nowbest/kinetics-uav-at-ib-idc-0603-g7cs1lm9-ep=6.ckpt', map_location='cuda:0')["state_dict"]
model.load_state_dict(params, strict=False)
model.eval()


def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


cam = GradCAM(model=model,
              target_layers=[model.transformer2.blocks[-1].norm1],
              # 这里的target_layer要看模型情况，调试时自己打印下model吧
              # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
              # 或者target_layers = [model.blocks[-1].ffn.norm]
              use_cuda=True,
              reshape_transform=reshape_transform)
image_path = "/data/liuxi/Dataset/k400-2-UAV_Human/crop_human/train/eat_snacks/P079S00G10B00H00UC022000LC021000A001R0_09171513/img_00008.jpg"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
target_category = None  # 可以指定一个类别，或者使用 None 表示最高概率的类别
input_tensor=torchvision.transforms.functional.to_tensor(rgb_img)
input_tensor=torchvision.transforms.functional.normalize(input_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32)/255,grayscale_cam)
# flops, params = profile(model, inputs=((img,label,img,label),), verbose = False)
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')


