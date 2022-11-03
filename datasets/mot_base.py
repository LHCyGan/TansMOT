"""MOT17 Dataset."""
import os

import numpy as np
import pandas as pd
import cv2
import pickle

from util.mox_env import wrap_input_path2


class MOTSingleBase:
    """Single folder reader for MOT and detection."""

    def __init__(
        self, data_folder_single, min_vis=0.1, vid_set="train", read_img=False
    ):
        self.vid_set = vid_set
        self.folder = data_folder_single
        self.read_img = read_img

        (
            self.vid_name,
            self.img_dir,
            self.seq_len,
            self.frame_rate,
            self.ori_size,
        ) = self._parse_info()

        self.data_vid_set = self.vid_set
        if self.vid_set == "train":
            (
                self._frame_id2box,
                self._frame2id,
                self._id2frame,
                self._frame_id2vis,
            ) = self._parse_gt()
        else:
            self._frame2box, self._frame2vis = self._parse_det(min_vis)

    def _parse_info(self):
        info_path = os.path.join(self.folder, "seqinfo.ini")
        if not os.path.exists(info_path):
            raise FileNotFoundError("File does not exist: {}".format(info_path))
        info = open(info_path).readlines()
        info = dict((tuple(x.strip().split("=")) for x in info if "=" in x))

        ori_size = (int(info["imHeight"]), int(info["imWidth"]))

        return (
            info["name"],
            info["imDir"],
            int(info["seqLength"]),
            int(info["frameRate"]),
            ori_size,
        )

    def _parse_gt(self):
        cache_name = self.folder.strip("/").replace("/", "_") + ".pkl"
        cache_dir = os.path.join(".cache", "TRTR")
        # if os.path.exists(os.path.join(cache_dir, cache_name)):
        #     return pickle.load(open(os.path.join(cache_dir, cache_name), "rb"))

        gt_path = os.path.join(self.folder, "gt", "gt.txt")
        gt_path = wrap_input_path2(gt_path)

        gt_df = pd.read_csv(gt_path, header=None)

        # Class and visivility filters
        # gt_df = gt_df[(gt_df[6] == 1) & (gt_df[8] > min_vis)]
        gt_df = gt_df[(gt_df[6] == 1)]

        # Keep frame, id and bbox
        gt_df.loc[:, [4, 5]] += gt_df[[2, 3]].values

        # Clip outlier bbox
        gt_df.loc[:, [2, 4]] = np.clip(gt_df[[2, 4]].values, 0, self.ori_size[1] - 1)
        gt_df.loc[:, [3, 5]] = np.clip(gt_df[[3, 5]].values, 0, self.ori_size[0] - 1)

        # Zero-index frame
        gt_df[0] -= 1
        frame_id2box = {k: v.values[0, 2:6] for k, v in gt_df.groupby([0, 1])}
        frame_id2vis = {k: v.values[0, 8] for k, v in gt_df.groupby([0, 1])}

        gt_df = gt_df[[0, 1]]
        frame2id = {k: v.values[:, 1] for k, v in gt_df.groupby(0)}
        id2frame = {k: v.values[:, 0] for k, v in gt_df.groupby(1)}

        frame2id.update(
            {k: np.zeros(0) for k in range(self.seq_len) if k not in frame2id}
        )

        os.makedirs(cache_dir, exist_ok=True)
        pickle.dump(
            (frame_id2box, frame2id, id2frame),
            open(os.path.join(cache_dir, cache_name), "wb"),
        )

        return frame_id2box, frame2id, id2frame, frame_id2vis

    def _parse_det(self, min_vis):
        cache_name = self.folder.strip("/").replace("/", "_") + ".pkl"
        cache_dir = os.path.join(".cache", "TRTR")
        if os.path.exists(os.path.join(cache_dir, cache_name)):
            return pickle.load(open(os.path.join(cache_dir, cache_name), "rb"))

        det_path = os.path.join(self.folder, "det", "det.txt")
        det_path = wrap_input_path2(det_path)
        det_df = pd.read_csv(det_path, header=None)

        # Keep frame and bbox
        det_df.loc[:, [4, 5]] += det_df[[2, 3]].values

        # Clip outlier bbox
        det_df.loc[:, [2, 4]] = np.clip(det_df[[2, 4]].values, 0, self.ori_size[1] - 1)
        det_df.loc[:, [3, 5]] = np.clip(det_df[[3, 5]].values, 0, self.ori_size[0] - 1)

        # Zero-index frame
        det_df[0] -= 1
        frame2box = {k: v.values[:, 2:6] for k, v in det_df.groupby(0)}
        frame2vis = {k: v.values[:, 6] for k, v in det_df.groupby(0)}

        frame2box.update(
            {k: np.empty((0, 4)) for k in range(self.seq_len) if k not in frame2box}
        )
        frame2vis.update(
            {k: np.empty((0,)) for k in range(self.seq_len) if k not in frame2vis}
        )

        pickle.dump(
            (frame2box, frame2vis), open(os.path.join(cache_dir, cache_name), "wb"),
        )

        return frame2box, frame2vis

    def __len__(self):
        return self.seq_len

    def _read_img(self, frame_idx):
        if not self.read_img:
            return None
        img_name = "{:06d}.jpg".format(frame_idx + 1)

        img_path = os.path.join(self.folder, "img1", img_name)
        img_path = wrap_input_path2(img_path)

        img = cv2.imread(img_path)
        if img is None:
            print("Read image fail: {}".format(img_path))
            return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _get_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx + 1 > len(self):
            raise IndexError("Out of range")

        target = {
            "vid_name": self.vid_name,
            "frame_idx": frame_idx,
            "ori_size": self.ori_size,
        }

        img = self._read_img(frame_idx)

        if self.data_vid_set != "train":
            bboxes = self._frame2box[frame_idx]
            labels = np.array([1] * bboxes.shape[0], dtype=np.int64)

            target.update({"bboxes": bboxes, "labels": labels})

            return {"img": img}, target

        ids = np.array(self._frame2id[frame_idx], dtype=np.int64)
        labels = np.array([1] * len(ids), dtype=np.int64)
        bboxes = np.array(
            [self._frame_id2box[(frame_idx, id)] for id in ids], dtype=np.float32
        )

        target.update({"bboxes": bboxes, "labels": labels, "ids": ids})

        return {"img": img}, target
