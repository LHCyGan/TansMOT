# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import numpy as np
import cv2
import pandas as pd
from torch._C import dtype
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
from torch import Tensor
from datasets.transforms import build as build_transform

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

# if float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    # batch[0]: inputs list, batch[1]: target list
    inputs, targets = list(zip(*batch))
    batch_size = len(inputs)
    tube_len = inputs[0]["tubes"].size(0)
    n_tubes = [x["tubes"].size(1) for x in inputs]
    max_n_dets = max([len(x["dets"]) for x in inputs])
    total_n_tracks = sum(n_tubes)

    n_tube_mask = torch.ones((batch_size, max(n_tubes)), dtype=torch.bool)

    tubes = torch.zeros((tube_len, total_n_tracks, 4))
    tubes_vis = torch.zeros((tube_len, total_n_tracks))
    tubes_mask = torch.ones((tube_len, total_n_tracks), dtype=torch.bool)

    dets = torch.zeros((batch_size, max_n_dets, 4))
    dets_vis = torch.zeros((batch_size, max_n_dets))
    dets_mask = torch.ones((batch_size, max_n_dets), dtype=torch.bool)

    cur_n = 0
    for i, x in enumerate(inputs):
        n_tube = x["tubes"].size(1)
        n_tube_mask[i, :n_tube] = 0

        tubes[:, cur_n : (cur_n + n_tube)] = x["tubes"]
        tubes_vis[:, cur_n : (cur_n + n_tube)] = x["tubes_vis"]
        tubes_mask[:, cur_n : (cur_n + n_tube)] = x["tubes_mask"]

        cur_n += n_tube

        n_det = len(x["dets"])
        dets[i, :n_det] = x["dets"]
        dets_vis[i, :n_det] = x["dets_vis"]
        dets_mask[i, :n_det] = 0

    cont = torch.zeros((batch_size, max(n_tubes)), dtype=torch.bool)
    bboxes = torch.zeros((batch_size, max(n_tubes), 4))
    tubes_dets_label = torch.zeros((batch_size, max(n_tubes)), dtype=torch.long)
    for i, x in enumerate(targets):
        n_tube = x["cont"].size(0)
        cont[i, :n_tube] = x["cont"]
        bboxes[i, :n_tube] = x["bboxes"]
        tube_id = x["tubes_id"]
        det_id = inputs[i]["dets_id"]
        tube_idx, det_idx = torch.where(tube_id[:, None] == det_id[None, :])
        tubes_dets_label[i, tube_idx] = det_idx

    vid_name = [x["vid_name"] for x in targets]
    frame_idx = [x["frame_idx"] for x in targets]
    ori_size = [x["ori_size"] for x in targets]

    final_inputs = {
        "n_tubes": n_tubes,
        "n_tubes_mask": n_tube_mask,
        "tubes": tubes,
        "tubes_vis": tubes_vis,
        "tubes_mask": tubes_mask,
        "dets": dets,
        "dets_vis": dets_vis,
        "dets_mask": dets_mask,
    }

    if "img" in inputs[0]:
        final_inputs.update({
            "img": [d["img"] for d in inputs],
        })

    targets = {
        "vid_name": vid_name,
        "frame_idx": frame_idx,
        "ori_size": ori_size,
        "cont": cont,
        "bboxes": bboxes,
        "tubes_dets_label": tubes_dets_label,
    }

    return final_inputs, targets


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def barrier():
    if not is_dist_avail_and_initialized():
        return

    dist.barrier()


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


def vis_output(samples, results, targets, output_dir, score_thre):
    transform = build_transform("reverse")
    output_dir = os.path.join(output_dir, "vis_output")
    os.makedirs(output_dir, exist_ok=True)
    imgss = []
    for sample, result, target in zip(samples, results, targets):
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(2)
        imgs, targets = transform(sample, target)
        boxess, scoress = result["boxes"], result["scores"]
        boxess, scoress = boxess.cpu().numpy(), scoress.cpu().numpy()
        boxess = boxess[scoress > score_thre]
        color = np.random.randint(0, 256, (len(boxess), 3))
        boxess = np.swapaxes(boxess, 0, 1)
        for img, boxes in zip(imgs, boxess):
            for i, bbox in enumerate(boxes):
                bbox = tuple(int(x) for x in bbox)
                cv2.rectangle(img, bbox[:2], bbox[2:], color[i].tolist(), 2)

            imgss.append(img)

        assert False

    for i, img in enumerate(imgss):
        img_path = os.path.join(output_dir, "{}.jpg".format(i))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img)


def vis_seq(seq_file, data_dir, output_dir):
    seq_name = os.path.splitext(os.path.basename(seq_file))[0]
    if seq_name not in os.listdir(data_dir):
        train_seq = ["02", "04", "05", "09", "10", "11", "13"]
        test_seq = ["01", "03", "06", "07", "08", "12", "14"]
        seq_num = seq_name[6:8]
        if seq_num in train_seq:
            data_dir = os.path.join(data_dir, "train")
        else:
            data_dir = os.path.join(data_dir, "test")

    imgs_dir = os.path.join(data_dir, seq_name, "img1")

    if not os.path.exists(imgs_dir):
        raise FileNotFoundError(imgs_dir)
    if not os.path.exists(seq_file):
        raise FileNotFoundError(seq_file)

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "{}.avi".format(seq_name))

    seq_len = len(os.listdir(imgs_dir))
    df = pd.read_csv(seq_file, header=None)
    df = df[df.columns[:6]]
    df.loc[:, [4, 5]] += df[[2, 3]].values

    df[0] -= 1
    frame_id2box = {k: v.values[0, 2:] for k, v in df.groupby([0, 1])}
    df = df[[0, 1]]
    # Zero-index frame
    frame2id = {k: v.values[:, 1] for k, v in df.groupby(0)}
    id2frame = {k: v.values[:, 0] for k, v in df.groupby(1)}

    frame2id.update({k: np.zeros(0) for k in range(seq_len) if k not in frame2id})

    color_map = {k: np.random.randint(0, 256, 3).tolist() for k in id2frame.keys()}

    writer = None
    for frame_idx, ids in tqdm(frame2id.items()):
        img = cv2.imread(os.path.join(imgs_dir, "{:06d}.jpg".format(frame_idx + 1)))

        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

        for track_id in ids:
            bbox = frame_id2box[(frame_idx, track_id)]
            bbox = tuple(int(x) for x in bbox)
            cv2.rectangle(img, bbox[:2], bbox[2:], color_map[track_id], 2)

        writer.write(img)

    writer.release()


class TensorboardLogger(SummaryWriter):
    def __init__(self, logdir=None, *args, **kwargs):
        if logdir is not None and logdir.startswith('s3://'):
            self.remote_logdir = logdir
            logdir = '/cache/tensorboard'
            os.makedirs(logdir, exist_ok=True)
        super(TensorboardLogger, self).__init__(logdir, *args, **kwargs)
        self.local_logdir = logdir

    def sync(self):
        if hasattr(self, 'remote_logdir'):
            mox.file.copy_parallel(
                self.local_logdir, self.remote_logdir, is_processing=False
            )


ARGS_TO_UPDATE = {
    "batch_size",
    "dataset_file",
    "mot17_root",
    "mot20_root",
    "mot17_feat_path",
    "mot20_feat_path",
    "pretrained_feat_extractor",
    "seqs",
    "match_thre",
    "testset",
    "output_dir",
    "eval",
    "vis_in_eval",
    "resume",
    "min_vis",
    "keep_pred",
    "seed",
    "distributed",
    "num_workers",
}
