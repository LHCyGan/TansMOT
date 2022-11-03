# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from scheduler import GradualWarmupScheduler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch, evaluate_val
from models import build_model

from collections import OrderedDict
from util.mox_env import wrap_input_path2

is_mox = False


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--lr_drop", default=900, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # * Model parameters

    # * Backbone
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "repeat"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument("--num_track_encoder_layers", default=6, type=int)
    parser.add_argument("--num_det_encoder_layers", default=3, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)

    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=128,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--no_pred_only",
        dest="pred_only",
        action="store_false",
        help="Use tube encoder for prediction directly",
    )

    # * Loss coefficients
    parser.add_argument("--bce_loss_coef", default=500, type=float)
    parser.add_argument("--match_loss_coef", default=0, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="mot17_feat_tube")
    parser.add_argument("--mot17_root", type=str, default="../../Dataset/MOT17/")
    parser.add_argument("--mot20_root", type=str, default="../../Dataset/MOT20/")
    parser.add_argument(
        "--mot17_feat_path",
        default="/Disk2/liyizhuo/TrackTrans/train_features_mot17.pkl",
        type=str,
    )
    parser.add_argument(
        "--mot20_feat_path",
        default="/Disk2/liyizhuo/TrackTrans/train_features_mot20.pkl",
        type=str,
    )
    parser.add_argument(
        "--mot17_test_feat_path", default="/Disk2/liyizhuo/TrackTrans/test_features.pkl"
    )
    parser.add_argument("--extract_feat", default='fasterrcnn', action="store_true")
    parser.add_argument(
        "--pretrained_feat_extractor",
        type=str,
        default="./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    )
    parser.add_argument("--tube_len", default=8, type=int)
    parser.add_argument("--keep_pred", default=0, type=int)
    parser.add_argument("--feat_dim", default=256, type=int)
    parser.add_argument("--seqs", "--names-list", nargs="+", default=None)

    parser.add_argument("--min_vis", default=0.25, type=float)
    parser.add_argument("--match_thre", default=0.0, type=float)
    parser.add_argument("--match_coef", default=0.0, type=float)
    parser.add_argument(
        "--testset", default="test", type=str, choices=("evaltrain", "test")
    )

    parser.add_argument("--drop_prob", default=0.0, type=float)

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--vis_in_eval", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--load_log_cfg", action="store_true")

    return parser


def main(args):
    # pretrained_path = wrap_input_path2(
    #     "s3://bucket-5006/lyz/pretrained/resnet50-19c8e357.pth"
    # )

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if ("backbone" not in n and p.requires_grad)
                or ("dim_reduce" in n and p.requires_grad)
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad and "dim_reduce" not in n
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    # step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=step_lr_scheduler)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(vid_set="train", args=args)
    dataset_val = build_dataset(vid_set="val", args=args)

    if args.eval:
        dataset_test = build_dataset(vid_set=args.testset, args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if args.eval:
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if args.eval:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=True,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    if args.eval:
        data_loader_test = DataLoader(
            dataset_test,
            args.batch_size,
            sampler=sampler_test,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    output_dir = Path(args.output_dir)
    if is_mox:
        output_dir = Path("/cache/TRTR/output")

    if args.output_dir and utils.is_main_process() and not args.eval:
        tensorboard_logger = utils.TensorboardLogger(args.output_dir)
        with (output_dir / "log" / "log.txt").open("a") as f:
            f.write(json.dumps(vars(args)) + "\n")
    else:
        tensorboard_logger = None

    if args.resume:
        if is_mox:
            assert not args.resume.startswith("https")
            local_path = os.path.join(
                "/cache/TRTR/pretrained/", os.path.basename(args.resume)
            )
            if utils.is_main_process():
                os.makedirs("/cache/TRTR/pretrained/", exist_ok=True)
                mox.file.copy(args.resume, local_path)

            args.resume = local_path

        utils.barrier()

        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     for ckpg, orpg in zip(checkpoint['optimizer']['param_groups'], optimizer.state_dict()['param_groups']):
        #         ckpg['lr'] = orpg['lr']

        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     checkpoint['lr_scheduler'] = {k: v for k, v in checkpoint['lr_scheduler'].items() if k != 'step_size'}
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        evaluate(
            model,
            # data_loader_train,
            data_loader_test,
            device,
            args.output_dir,
            args.tube_len,
            args.testset,
            args.vis_in_eval,
            args.match_thre,
            args.keep_pred,
            args.match_coef,
        )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            tensorboard_logger,
        )
        lr_scheduler.step()
        val_stats = evaluate_val(
            model,
            criterion,
            data_loader_val,
            device,
            epoch,
            args.clip_max_norm,
            tensorboard_logger,
        )
        if args.output_dir:
            # checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            if (
                (epoch + 1) % args.lr_drop == 0
                or (epoch + 1) % 1000 == 0
                or (epoch + 1) % args.epochs == 0
            ):
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                model_state_dict = model_without_ddp.state_dict()
                model_state_dict = {
                    k: v
                    for k, v in model_state_dict.items()
                    if "feat_extractor" not in k
                }
                utils.save_on_master(
                    {
                        "model": model_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
                if is_mox and utils.is_main_process():
                    mox.file.copy_parallel(
                        str(checkpoint_path),
                        os.path.join(
                            str(args.output_dir), os.path.basename(checkpoint_path)
                        ),
                    )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log" / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if is_mox and utils.is_main_process():
                mox.file.copy_parallel(
                    str(output_dir / "log" / "log.txt"),
                    os.path.join(args.output_dir, "log", "log.txt"),
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args, _ = parser.parse_known_args()
    utils.init_distributed_mode(args)

    if args.output_dir:
        # Check output dir
        if utils.is_main_process() and not args.eval and not args.resume:
            assert not Path(
                args.output_dir
            ).exists(), "Output dir already exists: {}".format(args.output_dir)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / "log").mkdir(exist_ok=True)

        # Load config from log
        if args.load_log_cfg:
            assert (
                Path(args.output_dir) / "log" / "log.txt"
            ).exists(), "Log file does not exist: {}".format(args.output_dir)
            with (Path(args.output_dir) / "log" / "log.txt").open() as f:
                old_args = json.loads(f.readline())

            args = vars(args)
            args_to_update = {
                k: args[k] for k in utils.ARGS_TO_UPDATE if k in utils.ARGS_TO_UPDATE
            }
            args_to_keep = {k: v for k, v in args.items() if k not in old_args}
            old_args.update(args_to_update)
            old_args.update(args_to_keep)

            args = argparse.Namespace(**old_args)

    main(args)
