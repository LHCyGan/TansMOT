"""MOT17 Dataset."""
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from datasets.mot_base import MOTSingleBase
from datasets.mot_det_transforms import build as build_transforms


class MOTDETSingle(MOTSingleBase):
    """Single folder reader for MOT and detection."""

    def __init__(self, data_folder_single, min_vis, vid_set):
        super(MOTDETSingle, self).__init__(
            data_folder_single, min_vis, vid_set, read_img=True
        )

    def __len__(self):
        return self.seq_len

    def __getitem__(self, frame_idx):
        inputs, target = super(MOTDETSingle, self)._get_frame(frame_idx)

        return inputs, target


class MOTDET(Dataset):
    """MOT17 Dataset."""

    def __init__(self, data_root, min_vis, transform, vid_set, seqs=None):
        folders = []
        for single_folder in os.listdir(data_root):
            if "MOT17" in single_folder and "FRCNN" not in single_folder:
                continue
            folders.append(os.path.join(data_root, single_folder))

        if seqs is not None:
            folders = [os.path.join(data_root, seq) for seq in seqs]

        self.seqs = [
            MOTDETSingle(single_folder, min_vis, vid_set,) for single_folder in folders
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


def build(vid_set, args):
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

    return MOTDET(
        os.path.join(args.mot17_root, data_vid_set),
        args.min_vis,
        build_transforms(trans_vid_set, args),
        data_vid_set,
        args.seqs,
    )

def build_mot20(vid_set, args):
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

    return MOTDET(
        os.path.join(args.mot20_root, data_vid_set),
        args.min_vis,
        build_transforms(trans_vid_set, args),
        data_vid_set,
        args.seqs,
    )
