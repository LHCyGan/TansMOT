from datasets.transforms import Expand
import cv2
import torch
from torch.distributed import constants
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF
import numpy as np

from PIL import Image
from matplotlib import cm

from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, targets):
        for t in self.transforms:
            inputs, targets = t(inputs, targets)
        return inputs, targets


class ConvertFromInts(object):
    def __call__(self, inputs, targets):
        for k in ["tubes", "dets", "tubes_vis", "dets_vis"]:
            inputs[k] = inputs[k].astype(np.float32)
        inputs["tubes_mask"] = inputs["tubes_mask"].astype(np.bool)
        if "img" in inputs:
            inputs["img"] = [
                Image.fromarray(img.astype(np.uint8)) for img in inputs["img"]
            ]
        targets["bboxes"] = targets["bboxes"].astype(np.float32)
        targets["cont"] = targets["cont"].astype(np.int64)
        return inputs, targets


class RandomDrop(object):
    def __init__(self, drop_prob=0.0):
        self.drop_prob = drop_prob

    def __call__(self, inputs, targets):
        drop_mask = np.random.rand(*inputs["tubes_mask"].shape) < self.drop_prob
        inputs["tubes_mask"] = np.logical_or(inputs["tubes_mask"], drop_mask)
        valid_mask = (~inputs["tubes_mask"]).any(0)
        for k in ["tubes_mask", "tubes", "tubes_vis"]:
            inputs[k] = inputs[k][:, valid_mask]
        for k in ["cont", "bboxes", "tubes_id"]:
            targets[k] = targets[k][valid_mask]
        return inputs, targets


class ToTensor(object):
    def __call__(self, inputs, targets):
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                inputs[k] = torch.from_numpy(v)
        if "img" in inputs:
            inputs["img"] = [transforms.ToTensor()(img) for img in inputs["img"]]

        for k, v in targets.items():
            if isinstance(v, np.ndarray):
                targets[k] = torch.from_numpy(v)

        return inputs, targets


class ToPercentCoords(object):
    def __call__(self, inputs, targets):
        h, w = targets["ori_size"]
        img_shape = np.array([w, h, w, h])
        inputs["tubes"] /= img_shape
        inputs["dets"] /= img_shape
        targets["bboxes"] /= img_shape

        inputs["tubes"] = box_xyxy_to_cxcywh(inputs["tubes"])
        inputs["dets"] = box_xyxy_to_cxcywh(inputs["dets"])
        targets["bboxes"] = box_xyxy_to_cxcywh(targets["bboxes"])

        return inputs, targets


class Resize(object):
    def __call__(self, inputs, targets):
        if "img" not in inputs:
            return inputs, targets
        h, w = targets["ori_size"]
        assert w >= h, "Width should be larger than height"
        tgt_w = w
        if w > 1333:
            tgt_w = 1333
        elif w < 800:
            tgt_w = 800
        tgt_h = int(h * tgt_w / w)
        inputs["img"] = [TF.resize(img, (tgt_h, tgt_w)) for img in inputs["img"]]

        return inputs, targets


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, inputs, targets):
        transform = self.color_jitter.get_params(
            self.color_jitter.brightness,
            self.color_jitter.contrast,
            self.color_jitter.saturation,
            self.color_jitter.hue,
        )
        inputs["img"] = [self.color_jitter(img) for img in inputs["img"]]

        return inputs, targets


class RandomFlip(object):
    def __call__(self, inputs, targets):
        """
        Bouding boxes should be formatted as [cx, cy, w, h]
        """
        if np.random.randint(2):
            inputs["img"] = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in inputs["img"]
            ]
            inputs["tubes"][..., 0] = 1 - inputs["tubes"][..., 0]
            inputs["dets"][..., 0] = 1 - inputs["dets"][..., 0]
            targets["bboxes"][..., 0] = 1 - targets["bboxes"][..., 0]

        return inputs, targets


class BoxNoise(object):
    def __add_noise(self, bboxes):
        x_c, y_c, w, h = np.split(bboxes, 4, -1)

        x_c += (np.random.rand(*w.shape) * 0.2 - 0.1) * w
        y_c += (np.random.rand(*h.shape) * 0.2 - 0.1) * h

        w *= 0.9 + 0.2 * np.random.rand(*w.shape)
        h *= 0.9 + 0.2 * np.random.rand(*h.shape)

        return np.concatenate([x_c, y_c, w, h], axis=-1)

    def __call__(self, inputs, targets):
        """
        Bouding boxes should be formatted as [cx, cy, w, h]
        """
        inputs["tubes"] = self.__add_noise(inputs["tubes"])
        inputs["dets"] = self.__add_noise(inputs["dets"])

        return inputs, targets


class MinVis(object):
    def __init__(self, min_vis=None):
        self.min_vis = min_vis

    def __call__(self, inputs, targets):
        if self.min_vis is None:
            lower = 0.1
            upper = 0.5
            min_vis = lower + np.random.rand() * (upper - lower)

            min_vis = (
                min((min_vis, np.max(inputs["dets_vis"]), np.max(inputs["tubes_vis"])))
                - 0.001
            )

        else:
            min_vis = self.min_vis


        det_valid_mask = inputs["dets_vis"] > min_vis
        for k in ["dets", "dets_id", "dets_vis"]:
            inputs[k] = inputs[k][det_valid_mask]

        tubes_valid_mask = inputs["tubes_vis"] > min_vis
        inputs["tubes_mask"] = np.logical_or(~tubes_valid_mask, inputs["tubes_mask"])
        valid_mask = (~inputs["tubes_mask"]).any(0)

        for k in ["tubes_mask", "tubes", "tubes_vis"]:
            inputs[k] = inputs[k][:, valid_mask]
        for k in ["cont", "bboxes", "tubes_id"]:
            targets[k] = targets[k][valid_mask]

        return inputs, targets


class ExpandAndCrop(object):
    def __sample_single_image(self, img, ratio, left_ratio, top_ratio):
        ori_img = np.array(img)
        ori_h, ori_w, c = ori_img.shape
        h, w = int(ori_h * (1 + ratio)), int(ori_w * (1 + ratio))
        t, l = int(ori_h * top_ratio), int(ori_w * left_ratio)
        img = np.zeros((h, w, c), dtype=ori_img.dtype)
        if ratio < 0:
            t, l = -t, -l
            img = ori_img[t : (t + h), l : (l + w)]
        else:
            img[t : (t + ori_h), l : (l + ori_w)] = ori_img

        return Image.fromarray(img)

    def __sample(self, inputs, targets, ratio):
        ratio = np.random.rand() * ratio
        left = np.random.rand() * ratio
        top = np.random.rand() * ratio
        # Expand bboxes
        for k in ["tubes", "dets"]:
            inputs[k][..., 0] += left
            inputs[k][..., 1] += top
            inputs[k] /= 1 + ratio

        targets["bboxes"][..., 0] += left
        targets["bboxes"][..., 1] += top
        targets["bboxes"] /= 1 + ratio
        # Expand images
        inputs["img"] = [
            self.__sample_single_image(img, ratio, left, top) for img in inputs["img"]
        ]

        return inputs, targets

    def __call__(self, inputs, targets):
        """
        Bouding boxes should be formatted as [cx, cy, w, h]
        """
        inputs, targets = self.__sample(inputs, targets, 0.2)
        inputs, targets = self.__sample(inputs, targets, -0.15)

        checker = CheckBox()
        while True:
            try:
                checker(inputs, targets)
            except AssertionError:
                continue
            break

        return inputs, targets


class CheckBox(object):
    def __check_box(self, bboxes):
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes[..., [0, 1]] = np.maximum(bboxes[..., [0, 1]], 0.0)
        bboxes[..., [2, 3]] = np.minimum(bboxes[..., [2, 3]], 1.0)
        bboxes = box_xyxy_to_cxcywh(bboxes)

        return bboxes

    def __valid_mask(self, bboxes):
        return np.logical_and(bboxes[..., 2] > 0.0, bboxes[..., 3] > 0.0)

    def __remove_invald(self, inputs, targets):
        det_valid_mask = self.__valid_mask(inputs["dets"])
        for k in ["dets", "dets_id", "dets_vis"]:
            inputs[k] = inputs[k][det_valid_mask]

        bbox_valid_mask = self.__valid_mask(targets["bboxes"])

        tubes_valid_mask = self.__valid_mask(inputs["tubes"])
        inputs["tubes_mask"] = np.logical_or(~tubes_valid_mask, inputs["tubes_mask"])
        valid_mask = (~inputs["tubes_mask"]).any(0)
        valid_mask = np.logical_and(valid_mask, bbox_valid_mask)

        for k in ["tubes_mask", "tubes", "tubes_vis"]:
            inputs[k] = inputs[k][:, valid_mask]
        for k in ["cont", "bboxes", "tubes_id"]:
            targets[k] = targets[k][valid_mask]

        return inputs, targets

    def __call__(self, inputs, targets):
        """
        This should be placed exactly before ToTensor()
        Bouding boxes should be formatted as [cx, cy, w, h]
        """
        inputs["tubes"] = self.__check_box(inputs["tubes"])
        inputs["dets"] = self.__check_box(inputs["dets"])
        targets["bboxes"] = self.__check_box(targets["bboxes"])

        self.__remove_invald(inputs, targets)

        assert (~inputs["tubes_mask"]).any(), "No tubes"
        assert len(inputs["dets"]) > 0, "No detections"
        assert (~inputs["tubes_mask"]).any(0).all(), "Empty tubes"

        return inputs, targets


class MOT17FeatTransforms(object):
    def __init__(self, vid_set, args):
        if vid_set == "train":
            self.augment = Compose(
                [
                    ConvertFromInts(),
                    ToPercentCoords(),
                    RandomFlip(),
                    ColorJitter(),
                    ExpandAndCrop(),
                    Resize(),
                    BoxNoise(),
                    MinVis(),
                    CheckBox(),
                    ToTensor(),
                ]
            )
        elif vid_set == "test" or vid_set == "val":
            self.augment = Compose(
                [ConvertFromInts(), ToPercentCoords(), MinVis(args.min_vis), ToTensor()]
            )
        else:
            raise NameError(
                "Wrong dataset type, should be in (train, test, val) :{}".format(
                    vid_set
                )
            )

    def __call__(self, imgs, targets):
        return self.augment(imgs, targets)


def build(vid_set, args):
    """Build Transformation
    Args:
        vid_set (str): Dataset type in train, test or reverse
        config (dict): Dataset configs
    Returns:
        MOT17Transforms: Transformation for MOT17 dataset
    """
    return MOT17FeatTransforms(vid_set, args)
