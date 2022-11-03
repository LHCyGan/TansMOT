"""MOT17 Dataset."""
import os

import argparse
import numpy as np
import pickle
from pandas.core import frame

from torch.utils.data import Dataset

from datasets.mot_base import MOTSingleBase
from datasets.mot_feat_tube_transforms import build as build_transforms


class MOTFeatSingle(MOTSingleBase):
    """Single folder reader for MOT and detection."""

    def __init__(
        self, data_folder_single, min_vis, vid_set, feat_path, tube_len, feat_dim
    ):
        super().__init__(data_folder_single, min_vis, vid_set, read_img=True)
        # self._frame_id2feat = self._load_feat(feat_path)
        self.tube_len = tube_len
        self.feat_dim = feat_dim

    # def _load_feat(self, feat_path):
    #     with open(feat_path, "rb") as f:
    #         feat = pickle.load(f)
    #     assert self.vid_name in feat, "Video {} not found in feature {}.".format(
    #         self.vid_name, feat_path
    #     )
    #     frame_id2feat = feat[self.vid_name]
    #     del feat
    #     return frame_id2feat

    def __len__(self):
        return super().__len__() - self.tube_len

    def __getitem__(self, frame_idx):
        frame_idxes = range(frame_idx, frame_idx + self.tube_len)
        target_frame_idx = frame_idx + self.tube_len
        if self.vid_set == "train":
            # Find ids of current frame
            dets_id = self._frame2id[target_frame_idx]
            # Dets in current frame
            dets = [
                self._frame_id2box.get((target_frame_idx, id), np.zeros(4))
                for id in dets_id
            ]
            # dets_feat = [
            #     self._frame_id2feat.get((target_frame_idx, id), np.zeros(self.feat_dim))
            #     for id in dets_id
            # ]
            dets_vis = [
                self._frame_id2vis.get((target_frame_idx, id), 0.0) for id in dets_id
            ]
            dets = np.array(dets)
            # dets_feat = np.array(dets_feat)
            dets_vis = np.array(dets_vis)

            inputs = {
                "dets": dets,  # [n_det, 4]
                # "dets_feat": dets_feat,  # [n_det, feat_dim]
                "dets_id": dets_id,  # [n_det]
                "dets_vis": dets_vis,  # [n_det]
            }
        else:
            inputs = {
                "dets": self._frame2box[frame_idx],
                # "dets_feat": self._frame_id2feat[frame_idx],
                "dets_vis": self._frame2vis[frame_idx],
                "dets_id": np.zeros(self._frame2box[frame_idx].shape[0]),
            }
            # assert len(inputs["dets"]) == len(inputs["dets_feat"]), (
            #     len(inputs["dets"]),
            #     len(inputs["dets_feat"]),
            # )

        target = {
            "vid_name": self.vid_name,
            "frame_idx": frame_idx,
            "ori_size": self.ori_size,
        }

        if self.read_img:
            # Reversed
            inputs.update({"img": [self._read_img(i) for i in frame_idxes[::-1]]})
            inputs["img"].append(self._read_img(target_frame_idx))

        if self.vid_set == "train" and self.tube_len > 0:
            # Find ids of all track
            tubes_id = self._frame2id[frame_idxes[-1]]
            # Find all tracks
            tubes_mask = [
                [0 if (i, id) in self._frame_id2box else 1 for id in tubes_id]
                for i in frame_idxes
            ]
            tubes = [
                [self._frame_id2box.get((i, id), np.zeros(4)) for id in tubes_id]
                for i in frame_idxes
            ]
            # tubes_feat = [
            #     [
            #         self._frame_id2feat.get((i, id), np.zeros(self.feat_dim))
            #         for id in tubes_id
            #     ]
            #     for i in frame_idxes
            # ]
            tubes_vis = [
                [self._frame_id2vis.get((i, id), 0.0) for id in tubes_id]
                for i in frame_idxes
            ]
            # latest bbox comes first
            tubes_mask = np.flip(np.array(tubes_mask, dtype=np.bool), axis=0)
            tubes = np.flip(np.array(tubes, dtype=np.float32), axis=0)
            # tubes_feat = np.flip(np.array(tubes_feat, dtype=np.float32), axis=0)
            tubes_vis = np.flip(np.array(tubes_vis, dtype=np.float32), axis=0)

            # Find boxes in current frame
            bboxes = np.array(
                [
                    self._frame_id2box.get((target_frame_idx, id), np.zeros(4))
                    for id in tubes_id
                ]
            )

            cont = np.array([1 if id in dets_id else 0 for id in tubes_id])

            inputs.update(
                {
                    "tubes_mask": tubes_mask,  # [tube_len, n_tube]
                    "tubes": tubes,  # [tube_len, n_tube, 4]
                    # "tubes_feat": tubes_feat,  # [tube_len, n_tube, feat_dim]
                    "tubes_vis": tubes_vis,  # [tube_len, n_tube]
                }
            )

            target.update(
                {
                    "cont": cont,  # [n_tube]
                    "bboxes": bboxes,  # [n_tube, 4]
                    "tubes_id": tubes_id,  # [n_tube]
                }
            )
        else:
            inputs.update(
                {
                    "tubes_mask": np.empty((0, 0)),  # [tube_len, n_tube]
                    "tubes": np.empty((0, 0, 4)),  # [tube_len, n_tube, 4]
                    # "tubes_feat": np.empty(
                        # (0, 0, self.feat_dim)
                    # ),  # [tube_len, n_tube, feat_dim]
                    "tubes_vis": np.empty((0, 0)),  # [tube_len, n_tube]
                }
            )

            target.update(
                {
                    "cont": np.empty(0),
                    "bboxes": np.empty((0, 4)),
                    "tubes_id": np.empty(0, dtype=np.int64),
                }
            )

        return inputs, target


class MOTFeat(Dataset):
    """MOT17 Dataset."""

    def __init__(
        self,
        data_root,
        min_vis,
        transform,
        vid_set,
        feat_path,
        tube_len,
        feat_dim,
        seqs=None,
    ):
        folders = []
        for single_folder in os.listdir(data_root):
            if "MOT17" in single_folder and "FRCNN" not in single_folder:
                continue
            folders.append(os.path.join(data_root, single_folder))

        if seqs is not None:
            folders = [os.path.join(data_root, seq) for seq in seqs]

        self.seqs = [
            MOTFeatSingle(
                single_folder, min_vis, vid_set, feat_path, tube_len, feat_dim
            )
            for single_folder in folders
        ]
        self._cum_lens = np.cumsum([len(seq) for seq in self.seqs])
        self._transform = transform

    def __len__(self):
        return self._cum_lens[-1]

    def __getitem__(self, index):
        if index < 0 or index + 1 > len(self):
            raise IndexError("Index {} out of length {}".format(index, len(self)))
        vid_idx = np.searchsorted(self._cum_lens, index, side="right")
        if vid_idx > 0:
            index -= self._cum_lens[vid_idx - 1]
        inputs, targets = self.seqs[vid_idx][index]
        if self._transform is not None:
            inputs, targets = self._transform(inputs, targets)

        return inputs, targets


class MOTFeatUnion:
    def __init__(self, seqs):
        self.seqs = seqs
        self._cum_lens = np.cumsum([len(d) for d in self.seqs])

    def __len__(self):
        return self._cum_lens[-1]

    def __getitem__(self, index):
        if index < 0 or index + 1 > len(self):
            raise IndexError("Index {} out of length {}".format(index, len(self)))
        vid_idx = np.searchsorted(self._cum_lens, index, side="right")
        if vid_idx > 0:
            index -= self._cum_lens[vid_idx - 1]
        inputs, targets = self.seqs[vid_idx][index]

        return inputs, targets


def build_mot17(vid_set, args):
    """Build dataset for MOT17.
    Args:
        vid_set (str): 'train', 'val' or 'test' set.
        configs (dict): Dataset configs.
    Raises:
        FileNotFoundError: Raise error if the root dir does not exist.
    Returns:
        MOT17: dataset for MOT17.
    """
    data_vid_set = trans_vid_set = vid_set
    if vid_set == "evaltrain":
        data_vid_set = "train"
        trans_vid_set = "test"
    tube_len = args.tube_len
    if vid_set == "evaltrain" or vid_set == "test":
        tube_len = 0

    return MOTFeat(
        os.path.join(args.mot17_root, data_vid_set),
        args.min_vis,
        build_transforms(trans_vid_set, args),
        data_vid_set,
        args.mot17_feat_path if not vid_set == "test" else args.mot17_test_feat_path,
        tube_len,
        args.feat_dim,
        args.seqs,
    )


def build_mot20(vid_set, args):
    data_vid_set = trans_vid_set = vid_set
    if vid_set == "evaltrain":
        data_vid_set = "train"
        trans_vid_set = "test"
    tube_len = args.tube_len
    if vid_set == "evaltrain" or vid_set == "test":
        tube_len = 0

    return MOTFeat(
        os.path.join(args.mot20_root, data_vid_set),
        args.min_vis,
        build_transforms(trans_vid_set, args),
        data_vid_set,
        args.mot20_feat_path,
        tube_len,
        args.feat_dim,
        args.seqs,
    )


def build_mot1720(vid_set, args):
    seqs = [build_mot17(vid_set, args), build_mot20(vid_set, args)]
    return MOTFeatUnion(seqs)


def build_val(args):
    # Sample MOT20-01 as validation
    return MOTFeat(
        os.path.join(args.mot20_root, "train"),
        args.min_vis,
        build_transforms("val", args),
        "train",
        args.mot20_feat_path,
        args.tube_len,
        args.feat_dim,
        ["MOT20-01"],
    )
