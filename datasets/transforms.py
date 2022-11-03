import torch
import cv2
import numpy as np
from numpy import random
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[..., 2:], box_b[2:])
    min_xy = np.maximum(box_a[..., :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[..., 0] * inter[..., 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


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

    def __call__(self, image, targets):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets


class ConvertFromInts(object):
    def __call__(self, imgs, targets):
        imgs = imgs.astype(np.float32)
        targets["tubes"] = targets["tubes"].astype(np.float32)
        return imgs, targets


class ConvertToInts(object):
    def __call__(self, imgs, targets):
        imgs = np.clip(imgs, 0.0, 255.0)
        imgs = imgs.astype(np.uint8)
        return imgs, targets


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, imgs, targets):
        imgs -= self.mean
        imgs /= 255.0
        return imgs, targets


class AddMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, imgs, targets):
        imgs *= 255
        imgs += self.mean
        return imgs, targets


class ToPercentCoords(object):
    def __call__(self, imgs, targets):
        _, height, width, _ = imgs.shape
        targets["tubes"][..., [0, 2]] /= width
        targets["tubes"][..., [1, 3]] /= height
        return imgs, targets


class ToAbsCoords(object):
    def __call__(self, imgs, targets):
        _, height, width, _ = imgs.shape
        targets["tubes"][..., [0, 2]] *= width
        targets["tubes"][..., [1, 3]] *= height
        return imgs, targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def _imrescale(self, img, scale, return_scale=False, interpolation="bilinear"):
        """Resize image while keeping the aspect ratio.
        Args:
            img (ndarray): The input image.
            scale (float or tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, then the image will
                be rescaled as large as possible within the scale.
            return_scale (bool): Whether to return the scaling factor besides the
                rescaled image.
            interpolation (str): Same as :func:`resize`.
        Returns:
            ndarray: The rescaled image.
        """

        def _scale_size(size, scale):
            """Rescale a size by a ratio.
            Args:
                size (tuple): w, h.
                scale (float): Scaling factor.
            Returns:
                tuple[int]: scaled size.
            """
            w, h = size
            return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

        def imresize(img, size, return_scale=False, interpolation="bilinear"):
            """Resize image to a given size.
            Args:
                img (ndarray): The input image.
                size (tuple): Target (w, h).
                return_scale (bool): Whether to return `w_scale` and `h_scale`.
                interpolation (str): Interpolation method, accepted values are
                    "nearest", "bilinear", "bicubic", "area", "lanczos".
            Returns:
                tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
                    `resized_img`.
            """
            interp_codes = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }

            h, w = img.shape[:2]
            resized_img = cv2.resize(
                img, size, interpolation=interp_codes[interpolation]
            )
            if not return_scale:
                return resized_img

            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

        h, w = img.shape[:2]
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError("Invalid scale {}, must be positive.".format(scale))
            scale_factor = scale
        elif isinstance(scale, (list, tuple)):
            scale_factor = min(scale[0] / h, scale[1] / w)
        else:
            raise TypeError(
                "Scale must be a number or tuple of int, but got {}".format(type(scale))
            )
        new_size = _scale_size((w, h), scale_factor)
        rescaled_img = imresize(img, new_size, interpolation=interpolation)
        if return_scale:
            return rescaled_img, scale_factor

        return rescaled_img

    def _impad(self, img, shape, pad_val=0):
        """Pad an image to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple): Expected padding shape.
            pad_val (number or sequence): Values to be filled in padding areas.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + [img.shape[-1]]
        assert len(shape) == len(img.shape)
        for i in range(len(shape) - 1):
            assert shape[i] >= img.shape[i]
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[: img.shape[0], : img.shape[1], ...] = img
        return pad

    def __call__(self, imgs, targets):
        h, w = imgs.shape[1:3]
        scale_factor = min(self.size[0] / h, self.size[1] / w)
        new_size = [int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)]
        h_pad_percent, w_pad_percent = (
            new_size[0] / self.size[0],
            new_size[1] / self.size[1],
        )
        imgs = [self._impad(self._imrescale(img, self.size), self.size) for img in imgs]

        # targets["tubes"][..., [0, 2]] *= w_pad_percent
        # targets["tubes"][..., [1, 3]] *= h_pad_percent
        imgs = np.array(imgs)
        targets["pad_percent"] = [w_pad_percent, h_pad_percent]

        return imgs, targets


class RandomSaturation(object):
    def __init__(self, lower=0.7, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            alpha = random.uniform(self.lower, self.upper)
            imgs[:, :, :, 1] = imgs[:, :, :, 1] * alpha

        return imgs, targets


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            delta = random.uniform(-self.delta, self.delta)
            imgs[:, :, :, 0] = imgs[:, :, :, 0] + delta
            imgs[:, :, :, 0][imgs[:, :, :, 0] > 360.0] -= 360.0
            imgs[:, :, :, 0][imgs[:, :, :, 0] < 0.0] += 360.0

        return imgs, targets


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            swap = self.perms[int(random.randint(len(self.perms)))]
            imgs = imgs[..., swap]
        return imgs, targets


class ConvertColor(object):
    def __init__(self, current="RGB", transform="HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, imgs, targets):
        if self.current == "RGB" and self.transform == "HSV":
            cvt = cv2.COLOR_RGB2HSV
        elif self.current == "HSV" and self.transform == "RGB":
            cvt = cv2.COLOR_HSV2RGB
        else:
            raise NotImplementedError
        imgs = np.array([cv2.cvtColor(img, cvt) for img in imgs])
        return imgs, targets


class RandomContrast(object):
    def __init__(self, lower=0.7, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            alpha = random.uniform(self.lower, self.upper)
            imgs = imgs * alpha
        np.clip(imgs, 0.0, 255.0)
        return imgs, targets


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            delta = random.uniform(-self.delta, self.delta)
            imgs = imgs + delta
        np.clip(imgs, 0.0, 255.0)
        return imgs, targets


class RandomSampleCrop(object):
    """Crop
    Arguments:
        mode (float tuple): the min and max jaccard overlaps
    """

    def crop(self, imgs, tubes, labels, w, h, left, top):
        crop_rect = np.array([int(left), int(top), int(left + w), int(top + h)])
        # cut the crop from the image
        imgs = imgs[:, crop_rect[1] : crop_rect[3], crop_rect[0] : crop_rect[2], :]

        overlap_area = jaccard(tubes, crop_rect)
        mask = overlap_area > 0
        labels[~mask] = 1

        # have any valid boxes? try again if not
        if not (labels != 1).all(-1).any():
            print("No valid")
            return None

        invalid_mask = (labels == 1).any(-1)


        # remove invalid tubes
        tubes = tubes[~invalid_mask]
        labels = labels[~invalid_mask]

        # should we use the box left and top corner or the crop's
        tubes[..., [0, 1]] = np.maximum(tubes[..., [0, 1]], crop_rect[[0, 1]])
        tubes[..., [2, 3]] = np.minimum(tubes[..., [2, 3]], crop_rect[[2, 3]])

        tubes -= crop_rect[[0, 1, 0, 1]]
        tubes[labels == 1, :] = 0.0

        return imgs, tubes, labels

    def __call__(self, imgs, targets):
        _, height, width, _ = imgs.shape

        while True:
            w = random.uniform(0.85 * width, width)
            h = random.uniform(0.85 * height, height)

            left = random.uniform(width - w)
            top = random.uniform(height - h)

            targets["ori_size"] = (h, w)
            res_pre = self.crop(
                imgs, targets["tubes"], targets["labels"], w, h, left, top
            )

            if res_pre is None:
                continue
            imgs, targets["tubes"], targets["labels"] = res_pre

            assert (targets["labels"] == 0).all()

            return imgs, targets


class Expand(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean

    def expand(self, imgs, ratio, left, top):
        n, h, w, c = imgs.shape
        new_imgs = np.zeros((n, int(h * ratio), int(w * ratio), c), dtype=imgs.dtype)
        new_imgs += self.mean
        new_imgs[:, top : (top + h), left : (left + w)] = imgs
        return new_imgs

    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            _, height, width, _ = imgs.shape
            ratio = random.uniform(1, 1.2)
            left = int(random.uniform(0, width * ratio - width))
            top = int(random.uniform(0, height * ratio - height))

            imgs = self.expand(imgs, ratio, left, top)
            targets["ori_size"] = (imgs.shape[1], imgs.shape[2])
            targets["tubes"] += np.array([int(left), int(top), int(left), int(top)])

        return imgs, targets


class RandomMirror(object):
    def __call__(self, imgs, targets):
        if random.rand() > 0.5:
            _, _, width, _ = imgs.shape
            imgs = imgs[..., ::-1, :]
            targets["tubes"][..., [0, 2]] = width - targets["tubes"][..., [2, 0]]

        return imgs, targets


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current="RGB", transform="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="RGB"),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, imgs, targets):
        imgs, targets = self.rand_brightness(imgs, targets)
        if random.rand() > 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        imgs, targets = distort(imgs, targets)

        return self.rand_light_noise(imgs, targets)


class ToTensor(object):
    def __call__(self, imgs, targets):
        imgs = torch.from_numpy(imgs).permute(3, 0, 1, 2)
        targets["tubes"] = box_xyxy_to_cxcywh(torch.from_numpy(targets["tubes"]))

        assert (targets["labels"] == 0).all()

        targets["labels"] = targets["labels"][..., 0]
        targets["labels"] = torch.from_numpy(targets["labels"])

        return imgs, targets


class ToArray(object):
    def __call__(self, imgs, targets):
        imgs = imgs.permute(1, 2, 3, 0).cpu().data.numpy()
        targets["tubes"] = (
            box_cxcywh_to_xyxy(targets["tubes"]).float().cpu().data.numpy()
        )
        targets["labels"] = targets["labels"].cpu().data.numpy()

        return imgs, targets


class MOT17Transforms(object):
    def __init__(self, size=[896, 1152], mean=[104, 117, 123], vid_set="train"):
        if vid_set == "train":
            self.augment = Compose(
                [
                    ConvertFromInts(),
                    PhotometricDistort(),
                    Expand(mean),
                    RandomSampleCrop(),
                    RandomMirror(),
                    ToPercentCoords(),
                    Resize(size),
                    SubtractMeans(mean),
                    ToTensor(),
                ]
            )
        elif vid_set == "reverse":
            self.augment = Compose(
                [ToArray(), AddMeans(mean), ToAbsCoords(), ConvertToInts()]
            )
        elif vid_set == "test":
            self.augment = Compose(
                [
                    ConvertFromInts(),
                    ToPercentCoords(),
                    Resize(size),
                    SubtractMeans(mean),
                    ToTensor(),
                ]
            )
        else:
            raise NameError(
                "Wrong dataset type, should be in (train, test) :{}".format(vid_set)
            )

    def __call__(self, imgs, targets):
        return self.augment(imgs, targets)


def build(vid_set):
    """Build Transformation

    Args:
        vid_set (str): Dataset type in train, test or reverse
        config (dict): Dataset configs

    Returns:
        MOT17Transforms: Transformation for MOT17 dataset
    """
    return MOT17Transforms(vid_set=vid_set)
