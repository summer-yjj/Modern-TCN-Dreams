import argparse
import random

import numpy as np
import torch

from exp.exp_pointseg_dreams200hz import Exp_PointSeg_Dreams200Hz
from utils.str2bool import str2bool


def build_parser():
    parser = argparse.ArgumentParser(description="ModernTCN point segmentation for dreams_scoring1_200hz")

    # basic
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="dreams200")
    parser.add_argument("--model", type=str, required=True, default="ModernTCN")
    parser.add_argument("--task_name", type=str, default="point_segmentation")

    # data
    parser.add_argument("--data", type=str, default="DREAMS200")
    parser.add_argument("--root_path", type=str, required=True, default="./all_datasets/dreams_scoring1_200hz")
    parser.add_argument("--num_workers", type=int, default=0)

    # model
    parser.add_argument("--ffn_ratio", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, nargs="+", default=[1])
    parser.add_argument("--large_size", type=int, nargs="+", default=[31])
    parser.add_argument("--small_size", type=int, nargs="+", default=[5])
    parser.add_argument("--dims", type=int, nargs="+", default=[128])
    parser.add_argument("--dw_dims", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--class_dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--use_multi_scale", type=str2bool, default=False)
    parser.add_argument("--small_kernel_merged", type=str2bool, default=False)

    # train
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--des", type=str, default="Exp")

    # point seg
    parser.add_argument("--pointseg_loss", type=str, default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--pointseg_balanced_sampling", type=str2bool, default=True)
    parser.add_argument("--pointseg_target_pos_ratio", type=float, default=0.25)
    parser.add_argument("--pointseg_pos_jitter_std", type=float, default=0.015)
    parser.add_argument("--pointseg_pos_jitter_prob", type=float, default=0.30)
    parser.add_argument("--pos_threshold", type=float, default=0.2)
    parser.add_argument("--pos_weight", type=float, default=8.0)
    parser.add_argument("--event_min_duration_sec", type=float, default=0.5)
    parser.add_argument("--event_merge_gap_sec", type=float, default=0.1)
    parser.add_argument("--event_max_duration_sec", type=float, default=3.0)
    parser.add_argument("--event_one_to_one", type=str2bool, default=True)
    parser.add_argument("--fs", type=float, default=200.0, help="sampling rate used by new preprocessed dataset")

    # compatibility args used by model/building code
    parser.add_argument("--enc_in", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=600)
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--test_flop", action="store_true", default=False)
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--subtract_last", type=int, default=0)
    parser.add_argument("--downsample_ratio", type=int, default=2)
    parser.add_argument("--stem_ratio", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--decomposition", type=int, default=0)
    parser.add_argument("--use_gpu_ids", type=str, default="")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--embed_type", type=int, default=0)
    parser.add_argument("--drop_last", type=str2bool, default=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_PointSeg_Dreams200Hz
    for ii in range(args.itr):
        setting = (
            f"{args.model_id}_{args.model}_{args.data}_"
            f"dim{args.dims[0]}_lk{args.large_size[0]}_sk{args.small_size[0]}_"
            f"psloss{args.pointseg_loss}_thr{args.pos_threshold}_it{ii}"
        )

        exp = Exp(args)
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
