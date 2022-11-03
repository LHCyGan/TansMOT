import torch
import numpy as np
import cv2


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
        inputs["img"] = inputs["img"].astype(np.float32)
        inputs["img"] /= 255
        targets["bboxes"] = targets["bboxes"].astype(np.float32)
        return inputs, targets


class Resize(object):
    def __call__(self, inputs, targets):
        h, w = targets["ori_size"]
        assert w >= h, "Width should be larger than height"
        tgt_w = w
        if w > 1333:
            tgt_w = 1333
        elif w < 800:
            tgt_w = 800
        tgt_h = int(h * tgt_w / w)
        inputs["img"] = cv2.resize(inputs["img"], (tgt_w, tgt_h))
        targets["bboxes"] *= np.array([tgt_w / w, tgt_h / h, tgt_w / w, tgt_h / h])

        return inputs, targets


class ToTensor(object):
    def __call__(self, inputs, targets):
        inputs["img"] = torch.from_numpy(inputs["img"]).permute(2, 0, 1)

        targets["labels"] = torch.from_numpy(targets["labels"])
        targets["bboxes"] = torch.from_numpy(targets["bboxes"])

        return inputs, targets


class MOT17_DETTransforms(object):
    def __init__(self, vid_set):
        if vid_set == "train":
            self.augment = Compose([ConvertFromInts(), Resize(), ToTensor()])
        elif vid_set == "test" or vid_set == "val":
            self.augment = Compose([ConvertFromInts(), Resize(), ToTensor()])
        else:
            raise NameError(
                "Wrong dataset type, should be in (train, test, val) :{}".format(
                    vid_set
                )
            )

    def __call__(self, imgs, targets):
        return self.augment(imgs, targets)


def build(vid_set, *args, **kwargs):
    """Build Transformation
    Args:
        vid_set (str): Dataset type in train, test or reverse
        config (dict): Dataset configs
    Returns:
        MOT17Transforms: Transformation for MOT17 dataset
    """
    return MOT17_DETTransforms(vid_set)
