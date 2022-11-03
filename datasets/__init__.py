# from .mot17 import build as build_mot
# from .mot17poi import build as build_mot17poi
# from .motdet import build as build_motdet
# from .motgt import build as build_motgt
from .mot_det import build as build_mot17_det
from .mot_det import build_mot20 as build_mot20_det
from .mot_feat_tube import build_mot17 as build_mot17_feat_tube
from .mot_feat_tube import build_mot20 as build_mot20_feat_tube
from .mot_feat_tube import build_mot1720 as build_mot1720_feat_tube
from .mot_feat_tube import build_val as build_val_feat_tube


DATASETS = {
    "mot17_det": build_mot17_det,
    "mot20_det": build_mot20_det,
    "mot17_feat_tube": build_mot17_feat_tube,
    "mot20_feat_tube": build_mot20_feat_tube,
    "mot1720_feat_tube": build_mot1720_feat_tube,
}

def build_dataset(vid_set, args):
    if vid_set == "val":
        return build_val_feat_tube(args)
    return DATASETS[args.dataset_file](vid_set, args)
    