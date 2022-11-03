# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from tqdm import tqdm
from typing import Iterable
from copy import deepcopy
import json

import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from scipy.optimize import linear_sum_assignment
from models.TrackTrans import FRCNNFeatureExtractor


def recur_to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, list):
        return [recur_to_cuda(y) for y in x]
    return x


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    tensorboard_logger=None,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = {k: recur_to_cuda(v) for k, v in samples.items()}
        targets = {k: recur_to_cuda(v) for k, v in targets.items()}

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if tensorboard_logger is not None:
            batch_size = data_loader.batch_sampler.batch_size
            step = (epoch * len(data_loader) + i) * batch_size
            step = round(step / 8)
            tensorboard_logger.add_scalar("scaled_loss/loss", loss_value, step)
            for k, v in loss_dict_reduced_unscaled.items():
                tensorboard_logger.add_scalar("unscaled_loss/{}".format(k), v, step)
            for k, v in loss_dict_reduced_scaled.items():
                tensorboard_logger.add_scalar("scaled_loss/{}".format(k), v, step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_val(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    tensorboard_logger=None,
):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = None
    print_freq = 1000000

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = {k: recur_to_cuda(v) for k, v in samples.items()}
        targets = {k: recur_to_cuda(v) for k, v in targets.items()}

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )

        if tensorboard_logger is not None:
            batch_size = data_loader.batch_sampler.batch_size
            step = (epoch * len(data_loader) + i) * batch_size
            step = round(step / 8)
            tensorboard_logger.add_scalar("val_scaled_loss/loss", loss_value, step)
            for k, v in loss_dict_reduced_unscaled.items():
                tensorboard_logger.add_scalar("val_unscaled_loss/{}".format(k), v, step)
            for k, v in loss_dict_reduced_scaled.items():
                tensorboard_logger.add_scalar("val_scaled_loss/{}".format(k), v, step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Evaluate average stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class Tracker(object):
    def __init__(self, ori_size, vid_name):
        """
        ori_size: (height, width)
        """
        self._tracks = {}
        self._tracks_mask = {}
        self._tracks_start = {}
        self._track_id = 1
        self._ori_size = ori_size
        self._vid_name = vid_name

    def add_new_tracks(self, tracks, tracks_mask, frame_idx):
        """
        tracks: (1, N, 4), torch.Tensor
        tracks_mask: (1, N), torch.Tensor
        """
        assert len(tracks) == 1
        tracks = tracks.cpu().squeeze(0)
        tracks_mask = tracks_mask.cpu().squeeze(0)
        new_tracks_id = []
        for box, mask in zip(tracks, tracks_mask):
            self._tracks[self._track_id] = {}
            if not mask:
                self._tracks[self._track_id][frame_idx] = box
            new_tracks_id.append(self._track_id)
            self._track_id += 1

        return new_tracks_id

    def update_tracks(self, bboxes, mask, tracks_id, frame_idx):
        """
        tracks: (1, N, 4), torch.Tensor
        tracks_mask: (1, N), torch.Tensor
        tracks_id: [N], list
        """
        bboxes = bboxes.squeeze(0).cpu()
        mask = mask.squeeze(0).cpu()
        assert len(bboxes) == len(mask) == len(tracks_id)
        for box, mask, track_id in zip(bboxes, mask, tracks_id):
            if not mask:
                self._tracks[track_id][frame_idx] = box

    def bbox_to_ori(self, bboxes, tgt_size=None):
        """Convert nomalized cxcywh to bounding boxes

        Args:
            bboxes (Tensor): [N, 4]
        """
        if tgt_size is None:
            height, width = self._ori_size
        else:
            height, width = tgt_size
        if bboxes.ndim == 1:
            bboxes = bboxes.unsqueeze(0)
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes *= torch.tensor(
            [[width, height, width, height]], dtype=bboxes.dtype, device=bboxes.device
        )
        return bboxes

    def write_to_file(self, output_dir, data_dir, vis=False):
        output_file = os.path.join(output_dir, "{}.txt".format(self._vid_name))
        height, width = self._ori_size
        with open(output_file, "w") as fp:
            for track_id, track in self._tracks.items():
                for frame_idx, box in track.items():
                    box = self.bbox_to_ori(box.unsqueeze(0))
                    box = box.squeeze(0).numpy()
                    box[2:] -= box[:2]
                    fp.write(
                        "{},{},{},{},{},{},-1,-1,-1,-1\n".format(
                            frame_idx + 1, track_id, *tuple(box)
                        )
                    )
        if vis:
            utils.vis_seq(output_file, data_dir, os.path.join(output_dir, "vis"))


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    output_dir,
    tube_len,
    test_set,
    vis_in_eval,
    match_thre,
    keep_pred=0,
    match_coef=0.0,
):
    assert keep_pred <= tube_len
    if not os.path.exists(output_dir):
        raise FileNotFoundError(output_dir)
    # feat_extractor = FRCNNFeatureExtractor(
    #     "/Disk2/liyizhuo/pretrained/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    # )

    # feat_extractor.to(device)
    # feat_extractor.eval()
    model.eval()

    if not os.path.basename(output_dir):
        output_dir = os.path.dirname(output_dir)

    data_dir = os.path.dirname(os.path.dirname(data_loader.dataset.seqs[0].folder))
    result_dir = os.path.join(output_dir, test_set)
    os.makedirs(result_dir, exist_ok=True)

    # In case of unbounded
    cur_vid, tracker = None, None
    cur_ids = []
    # (S, N, 4), (S, N, 128), (S, N)
    # cur_tubes, cur_tubes_feat, cur_tubes_mask = None, None, None
    cur_tubes, cur_tubes_mask = None, None

    masked_count = 0

    pbar = tqdm(data_loader)
    for samples, targets in pbar:
        # Prepare data
        info = {k: v[0] for k, v in targets.items()}

        # Move all tensors to device
        samples = {k: recur_to_cuda(v) for k, v in samples.items()}

        # Switching video
        if info["vid_name"] != cur_vid:
            if tracker is not None:
                tracker.write_to_file(result_dir, data_dir, vis=vis_in_eval)
            cur_vid = info["vid_name"]
            tracker = Tracker(info["ori_size"], info["vid_name"])
            cur_ids, cur_imgs = [], []
            # cur_tubes, cur_tubes_feat, cur_tubes_mask = None, None, None
            cur_tubes, cur_tubes_mask = None, None

        # Currently there is no tubes
        if len(cur_ids) == 0:
            cur_tubes = samples["dets"]
            # cur_tubes_feat = samples["dets_feat"]
            cur_tubes_mask = samples["dets_mask"]
            cur_ids = tracker.add_new_tracks(
                cur_tubes, cur_tubes_mask, info["frame_idx"]
            )
            cur_imgs = [samples["img"][0][0]]
            continue

        # Prediction
        # assert not cur_tubes_mask.all(0).any()
        # painted_cur_tubes_mask = deepcopy(cur_tubes_mask)
        # for i in range(painted_cur_tubes_mask.size(1)):
        #     for j in range(painted_cur_tubes_mask.size(0)):
        #         if painted_cur_tubes_mask[j, i]:
        #             painted_cur_tubes_mask[j, i] = 0
        #         else:
        #             break

        cur_imgs = cur_imgs[: cur_tubes.size(0)]
        cur_imgs.append(samples["img"][0][0])

        outputs = model(
            {
                "n_tubes": [cur_tubes.size(1)],
                "n_tubes_mask": torch.zeros(
                    (1, cur_tubes.size(1)), dtype=torch.bool
                ).to(cur_tubes.device),
                "tubes": cur_tubes,
                "tubes_mask": cur_tubes_mask,
                "dets": samples["dets"],
                "dets_mask": samples["dets_mask"],
                "img": [cur_imgs],
            }
        )

        try:
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        except KeyError:
            pred_boxes, match_logits = outputs["pred_boxes"], outputs["match_logits"]

        # Match pred_boxes with dets using iou and refine pred_boxes
        cost_iou = (
            -box_iou(
                box_cxcywh_to_xyxy(samples["dets"].squeeze(0)),
                box_cxcywh_to_xyxy(pred_boxes.squeeze(0)),
            )[0]
            .cpu()
            .numpy()
        )

        # match_logits /= 0.07
        cost_iou -= (
            match_coef
            * match_logits.squeeze(0).softmax(dim=1).transpose(0, 1).cpu().numpy()
        )

        indices = linear_sum_assignment(cost_iou)
        indices_matched = tuple([x[-cost_iou[indices] > match_thre] for x in indices])
        # pred_feat = torch.zeros_like(cur_tubes_feat[[0]])
        pred_mask = torch.ones_like(cur_tubes_mask[[0]])
        for i, j in zip(*indices_matched):
            pred_boxes[0, j] = samples["dets"][0, i]
            # pred_feat[0, j] = samples["dets_feat"][0, i]
            pred_mask[0, j] = 0

        # Tracks not matched is refined with ROI feature
        not_matched_tubes_inds = [
            i for i in range(cur_tubes.size(1)) if i not in indices_matched[1]
        ]
        matched_tubes_inds = [
            i for i in range(cur_tubes.size(1)) if i in indices_matched[1]
        ]
        if keep_pred > 0:
            # Find the first non-masked position
            # (S, N)
            _, idx = torch.min(cur_tubes_mask, dim=0)
            # Remove those masked too long
            not_matched_tubes_inds = [
                i for i in not_matched_tubes_inds if idx[i] < keep_pred
            ]
            bboxes = pred_boxes[0, not_matched_tubes_inds].squeeze(0)
            bboxes = [tracker.bbox_to_ori(bboxes, samples["img"][0][0].shape[1:])]
            # roi_features = feat_extractor(samples["img"][0], bboxes)
            # pred_feat[0, not_matched_tubes_inds] = roi_features
            # Count refined tracks as matched
            matched_tubes_inds += not_matched_tubes_inds

        # Update predicted bboxes
        tracker.update_tracks(pred_boxes, pred_mask, cur_ids, info["frame_idx"])

        cur_tubes = torch.cat([pred_boxes, cur_tubes], dim=0)
        # cur_tubes_feat = torch.cat([pred_feat, cur_tubes_feat], dim=0)
        cur_tubes_mask = torch.cat([pred_mask, cur_tubes_mask], dim=0)

        # Remove bboxes exceeding tube length
        cur_tubes = cur_tubes[:tube_len]
        # cur_tubes_feat = cur_tubes_feat[:tube_len]
        cur_tubes_mask = cur_tubes_mask[:tube_len]

        cur_tubes_len, cur_tubes_n = cur_tubes.shape[:2]

        # Keep matched tracks
        cur_tubes = cur_tubes[:, matched_tubes_inds]
        # cur_tubes_feat = cur_tubes_feat[:, matched_tubes_inds]
        cur_tubes_mask = cur_tubes_mask[:, matched_tubes_inds]
        cur_ids = [cur_ids[i] for i in matched_tubes_inds]

        # Dets not matched form new tracks
        not_matched_dets_inds = [
            i for i in range(samples["dets"].size(1)) if i not in indices_matched[0]
        ]

        n_new_tracks = len(not_matched_dets_inds)

        if n_new_tracks == 0:
            continue

        new_tracks = torch.zeros(
            [cur_tubes_len, n_new_tracks, 4], device=cur_tubes.device,
        )
        new_tracks[0] = samples["dets"][0, not_matched_dets_inds]

        # new_tracks_feat = torch.zeros(
        # [cur_tubes_len, n_new_tracks, cur_tubes_feat.size(2),],
        # device=cur_tubes_feat.device,
        # )
        # new_tracks_feat[0] = samples["dets_feat"][0, not_matched_dets_inds]

        new_tracks_mask = torch.ones(
            [cur_tubes_len, n_new_tracks],
            dtype=torch.bool,
            device=cur_tubes_mask.device,
        )
        new_tracks_mask[0] = samples["dets_mask"][0, not_matched_dets_inds]

        cur_tubes = torch.cat([cur_tubes, new_tracks], dim=1)
        # cur_tubes_feat = torch.cat([cur_tubes_feat, new_tracks_feat], dim=1)
        cur_tubes_mask = torch.cat([cur_tubes_mask, new_tracks_mask], dim=1)

        cur_ids += tracker.add_new_tracks(
            samples["dets"][:, not_matched_dets_inds],
            samples["dets_mask"][:, not_matched_dets_inds],
            info["frame_idx"],
        )

        pbar.set_description(
            "Cur tracks: {} Dets: {} New tracks: {}".format(
                cur_tubes.size(1), samples["dets"].size(1), n_new_tracks
            )
        )

    tracker.write_to_file(result_dir, data_dir, vis_in_eval)
