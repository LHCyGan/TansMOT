#%%
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from datasets import build_dataset
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

#%%

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretrain_path",
    default="/Disk2/liyizhuo/pretrained/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    type=str,
)

parser.add_argument("--min_vis", default=-0.1, type=float)
parser.add_argument("--mot17_root", default="/ssd2/data/MOT17", type=str)
parser.add_argument("--mot20_root", default="/ssd2/data/MOT20", type=str)
parser.add_argument(
    "--vid_set", default="evaltrain", type=str, choices=["evaltrain", "train", "test"]
)
parser.add_argument("--dataset_file", default="mot17_det", type=str)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--seqs", '--names-list', nargs='+', default=None)

args = parser.parse_args()  # running in command line
# args = parser.parse_args("")  # running in ipynb


# %% build dataset
dataset = build_dataset(args.vid_set, args)
sampler = torch.utils.data.SequentialSampler(dataset)
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler, args.batch_size, drop_last=False
)


def collate_fn(batch):
    inputs, targets = list(zip(*batch))
    inputs = {k: [d[k] for d in inputs] for k in inputs[0].keys()}
    targets = {k: [d[k] for d in targets] for k in targets[0].keys()}

    return inputs, targets


data_loader_train = DataLoader(
    dataset,
    batch_sampler=batch_sampler_train,
    collate_fn=collate_fn,
    num_workers=args.num_workers,
)

# %% build and load model
state_dict = torch.load(args.pretrain_path, map_location="cpu")
model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
model.load_state_dict(state_dict)
roi_pooler = MultiScaleRoIAlign(
    featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
)
avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

model.cuda()
roi_pooler.cuda()
avg_pool.cuda()

model.eval()
roi_pooler.eval()
avg_pool.eval()

# %%
ret = {}

for inputs, targets in tqdm(data_loader_train):
    # move to GPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.cuda()
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            inputs[k] = [vv.cuda() for vv in v]

    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            targets[k] = v.cuda()
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            targets[k] = [vv.cuda() for vv in v]
    # print(inputs["img"][0].shape)
    # print(inputs["img"][0].mean(), inputs["img"][0].std())
    # print(targets["bboxes"][0][0])
    with torch.no_grad():
        imgs, _ = model.transform(inputs["img"], None)
        features = model.backbone(imgs.tensors)
        roi_feats = roi_pooler(features, targets["bboxes"], imgs.image_sizes)
        roi_feats = avg_pool(roi_feats).squeeze()
    # print(roi_features[0].mean(), roi_features[0].std())
    # print(targets["vid_name"][0], targets["frame_idx"][0], targets["ids"][0][0])
    # assert False
    boxes_per_image = [bboxes.shape[0] for bboxes in targets["bboxes"]]
    roi_feats_per_img = roi_feats.cpu().split(boxes_per_image, 0)
    # count = 0
    for i, roi_features_img in enumerate(roi_feats_per_img):
        vid_name = targets["vid_name"][i]
        if vid_name not in ret:
            ret[vid_name] = {}
        frame_idx = targets["frame_idx"][i]
        if args.vid_set == "test":
            ret[vid_name][frame_idx] = roi_features_img.numpy()
        else:
            ids = targets["ids"][i]
            for roi_feat, id in zip(roi_feats_per_img[i], ids):
                ret[vid_name][(frame_idx, id)] = roi_feat.numpy()
                # if vid_name == "MOT20-01" and frame_idx == 0 and id == 1:
                #     assert False, (
                #         roi_feat.mean(),
                #         roi_feat.std(),
                #         roi_feat.numpy().mean(),
                #         roi_feat.numpy().std(),
                #         ret["MOT20-01"][(0, 1)].mean(),
                #         ret["MOT20-01"][(0, 1)].std(),
                #     )

with open("/Disk2/liyizhuo/TrackTrans/train_features_mot17.pkl", "wb") as f:
    # Pickle dictionary using protocol 0.
    pickle.dump(ret, f)

# roi_feat = ret["MOT20-01"][(0, 1)]
# assert False, (roi_feat.mean(), roi_feat.std())

# # %%
# import pickle
# feat = pickle.load(open("/Disk2/liyizhuo/TrackTrans/train_features.pkl", "rb"))
# # %%
# import numpy as np
# from tqdm import tqdm
# all_ret = []
# for vid in feat.values():
#     ret = None
#     for f in tqdm(vid.values()):
#         if ret is None:
#             ret = f[None, :]
#         else:
#             ret = np.concatenate([ret, f[None, :]], axis=0)
#     all_ret.append(ret)

# #%%
# final = np.concatenate(all_ret, axis=0)

# #%%
# mean = np.mean(final, axis=0)
# std = np.std(final, axis=0)

# %%
