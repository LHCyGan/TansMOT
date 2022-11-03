#%%
import math
import torch
from torch import nn
import torch.nn.functional as F
import sys
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign, roi_align, roi_pool
import time

sys.path.append("..")
from util import box_ops


class FRCNNFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        state_dict = torch.load(pretrain_path, map_location="cpu")
        self.net = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=False)
        self.net.load_state_dict(state_dict)
        # self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def __scale_bboxes(self, imgs, bboxes):
        bboxes_max = max([bbox.max() for bbox in bboxes if torch.numel(bbox) > 0])
        if bboxes_max <= 1.0:
            bboxes = [
                box_ops.unscale_box(box, wh[1], wh[0])
                for box, wh in zip(bboxes, imgs.image_sizes)
            ]

        return bboxes

    @torch.no_grad()
    def forward(self, imgs, bboxes):
        """
        Args:
            imgs (list): List of images, L * [C, H, W]
            bboxes (list): List of bboxes, L * [N, 4]

        Returns:
            Tensor: ROI features, sum(N for N in bboxes) * feat_dim
        """
        self.eval()
        imgs, _ = self.net.transform(imgs, None)
        bboxes = self.__scale_bboxes(imgs, bboxes)
        features = self.net.backbone(imgs.tensors)
        roi_features = roi_align(features["0"], bboxes, (1, 1), sampling_ratio=2)
        roi_features = roi_features.squeeze()
        return roi_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=8):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.flip(0).unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, x):
        assert x.size(0) <= self.max_len
        x = x + self.pe[: x.size(0), :, :]
        return x


class TrackTrans(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_track_encoder_layers=2,
        num_det_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        pred_only=False,
    ):
        super().__init__()
        self.track_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_track_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.det_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_det_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.d_model = d_model
        self.pred_only = pred_only

    def forward(self, tubes, tubes_mask, n_tubes, n_tubes_mask, dets, dets_mask):
        """
        Input Shape:
            - tubes: (St, Nt, E)
            - tubes_mask: (St, Nt)
            - n_tubes: (N,) sum(n_tubes) = Nt
            - n_tubes_mask: (N, max_n_tube)
            - dets: (N, max_n_det, E)
            - dets_mask: (N, max_n_det)
        Output Shape: (St, N, E)
        """
        assert sum(n_tubes) == tubes.size(1)

        # tubes: (St, Nt, E), tubes_mask: (St, Nt), track_queries: (St, Nt, E)
        tube_queries = self.track_encoder(
            tubes, src_key_padding_mask=tubes_mask.transpose(0, 1)
        )
        # (Nt, E)
        tube_queries = torch.sum(tube_queries * (~tubes_mask).unsqueeze(-1), dim=0)
        tube_queries = tube_queries / (~tubes_mask).sum(0).unsqueeze(-1)

        n_tube_max = max(n_tubes)
        # N x (n_track, E)
        tube_queries = tube_queries.split(n_tubes, dim=0)
        # (N, max_n_tube, E)
        tube_queries = torch.stack(
            [
                F.pad(q, (0, 0, 0, n_tube_max - n_track))
                for q, n_track in zip(tube_queries, n_tubes)
            ]
        )

        tube_queries = tube_queries.transpose(0, 1)

        # if self.pred_only:
        #     return tube_queries

        # dets: (N, max_n_det, E), dets_mask: (N, max_n_det), det_memory: (max_n_det, N, E)
        det_memory = self.det_encoder(
            dets.transpose(0, 1), src_key_padding_mask=dets_mask
        )

        # tube_queries: (max_n_tube, N, E), det_memory: (max_n_det, N, E)
        # n_tubes_mask: (N, max_n_tube), dets_mask: (N, max_n_det)
        # return: (max_n_tube, N, E)
        return self.decoder(tube_queries,memory=det_memory,
                tgt_key_padding_mask=n_tubes_mask,
                memory_key_padding_mask=dets_mask,), det_memory
        # decoder_fea = self.decoder(
        #     tube_queries, memory=det_memory,
        #     tgt_key_padding_mask=n_tubes_mask,
        #     memory_key_padding_mask=dets_mask
        # )
        # return



class TT(nn.Module):
    def __init__(self, pos_embed, transformer, feat_dim, feat_extractor=None):
        super().__init__()
        hidden_dim = transformer.d_model
        self.pos_embed = pos_embed
        self.transformer = transformer
        self.feat_dim = feat_dim
        self.input_proj = MLP(feat_dim + 4, hidden_dim, hidden_dim, 3)
        # self.class_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.feat_extractor = feat_extractor

    def extract_feat(self, samples):
        assert self.feat_extractor is not None
        # if self.feat_extractor is None:
        #     return samples
        imgs = samples["img"]
        n_tubes = samples["n_tubes"]
        n_tubes_mask = samples["n_tubes_mask"]
        tubes = samples["tubes"]
        tubes_mask = samples["tubes_mask"]
        dets = samples["dets"]
        dets_mask = samples["dets_mask"]

        tubes_feat = torch.zeros(*samples["tubes_mask"].shape, self.feat_dim).to(tubes.device)
        dets_feat = torch.zeros(*samples["dets_mask"].shape, self.feat_dim).to(dets.device)

        tubes = box_ops.box_cxcywh_to_xyxy(tubes)
        dets = box_ops.box_cxcywh_to_xyxy(dets)

        assert tubes.size(0) + 1 == len(imgs[0])
        # Flatten everything
        flat_imgs = [img for img_list in imgs for img in img_list[:-1]]
        # Append target frame at last
        flat_imgs = flat_imgs + [img_list[-1] for img_list in imgs]

        flat_tubes = [
            tube for tube_list in tubes.split(n_tubes, dim=1) for tube in tube_list
        ]
        flat_tubes_mask = [
            tube_mask
            for tube_mask_list in tubes_mask.split(n_tubes, dim=1)
            for tube_mask in tube_mask_list
        ]
        bboxes = [
            tube[~tube_mask] for tube, tube_mask in zip(flat_tubes, flat_tubes_mask)
        ]
        bboxes = bboxes + [det[~det_mask] for det, det_mask in zip(dets, dets_mask)]

        n_tubes_bboxes, n_dets = (~tubes_mask).sum(), (~dets_mask).sum()
        roi_feats = self.feat_extractor(flat_imgs, bboxes)
        tubes_feat[~tubes_mask] = roi_feats[:n_tubes_bboxes]
        dets_feat[~dets_mask] = roi_feats[n_tubes_bboxes:]

        samples["dets_feat"] = dets_feat
        samples["tubes_feat"] = tubes_feat
        return samples

    def forward(self, samples):
        """ samples is a dict, which consists of:
                - "n_tubes": (N,) sum(n_tubes) = Nt
                - "n_tube_mask": (N, max_n_tube)
                - "tubes": (St, Nt, 4)
                - "tubes_feat": (St, Nt, feat_dim)
                - "tubes_mask": (St, Nt)
                - "dets": (N, max_n_det, 4)
                - "dets_feat": (N, max_n_det, feat_dim)
                - "dets_mask": (N, max_n_det)
            outputs is a dict, which consists of:
                - "pred_logits": (N, max_n_tube)
                - "pred_boxes": (N, max_n_tube, 4)
        """
        samples = self.extract_feat(samples)

        n_tubes = samples["n_tubes"]
        n_tubes_mask = samples["n_tubes_mask"]
        tubes = samples["tubes"]
        tubes_feat = samples["tubes_feat"]
        tubes_mask = samples["tubes_mask"]
        dets = samples["dets"]
        dets_feat = samples["dets_feat"]
        dets_mask = samples["dets_mask"]

        # (St, Nt, feat_dim + 4) -> (St, Nt, E)
        tubes = self.input_proj(torch.cat([tubes, tubes_feat], dim=-1))
        # print(tubes.size())
        # (St, Nt, E) -> (St, Nt, E)
        tubes = self.pos_embed(tubes)
        # (N, max_n_det, 4), (N, max_n_det, feat_dim) -> (N, max_n_det, E)
        dets = self.input_proj(torch.cat([dets, dets_feat], dim=-1))

        # hidden_state: (max_n_tube, N, E) det_memory: (max_n_det, N, E)
        # print(len(self.transformer(
        #     tubes, tubes_mask, n_tubes, n_tubes_mask, dets, dets_mask
        # )))
        hidden_state, det_memory = self.transformer(
            tubes, tubes_mask, n_tubes, n_tubes_mask, dets, dets_mask
        )

        # hidden_state: (N, max_n_tube, E) det_memory: (N, max_n_det, E)
        hidden_state, det_memory = (
            hidden_state.transpose(0, 1),
            det_memory.transpose(0, 1),
        )

        assert not torch.isnan(hidden_state).any()

        tubes_embed = F.normalize(hidden_state, dim=-1)
        dets_embed = F.normalize(det_memory, dim=-1)
        # (N, max_n_tube, max_n_det)
        match_logits = torch.einsum("nie,nje->nij", tubes_embed, dets_embed)

        # # (N, max_n_tube)
        # outputs_class = self.class_embed(hidden_state).squeeze(-1)
        # (N, max_n_tube, 4)
        outputs_coord = self.bbox_embed(hidden_state).sigmoid()

        outputs = {
            "n_tubes_mask": n_tubes_mask, # (N, max_n_tube)
            "dets_mask": dets_mask, # (N, max_n_det)
            # "pred_logits": outputs_class,
            "match_logits": match_logits,  # (N, max_n_tube, max_n_det)
            "pred_boxes": outputs_coord,  # (N, max_n_tube, 4)
        }

        return outputs


class SetCriterion(nn.Module):
    def __init__(self, losses, weight_dict):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def loss_labels(self, outputs, targets):
        """ outputs consists of:
                - "n_tube_mask" (N, max_n_tube)
                - "pred_logits" (N, max_n_tube)
            targets consists of:
                - "cont" (N, max_n_tube)
        """
        n_tubes_mask = outputs["n_tubes_mask"]
        pred_logits = outputs["pred_logits"]
        labels = targets["cont"]

        # (N, max_n_tube)
        loss_bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            labels.float(),
            reduction="none",
            pos_weight=pred_logits.new([1 / 200]),
        )
        loss_bce = loss_bce[~n_tubes_mask].mean()

        pos_prop = (
            100 * labels[~n_tubes_mask].float().sum() / (~n_tubes_mask).float().sum()
        )

        return {"loss_bce": loss_bce, "pos_prop": pos_prop}

    def loss_match(self, outputs, targets):
        """ outputs consists of:
                - "n_tubes_mask" (N, max_n_tube)
                - "dets_mask" (N, max_n_det)
                - "match_logits" (N, max_n_tube, max_n_det)
            targets consists of:
                - "tubes_dets_label" (N, max_n_tube)
        """
        n_tube_mask = outputs["n_tubes_mask"]
        dets_mask = outputs["dets_mask"]
        match_logits = outputs["match_logits"]
        tubes_dets_label = targets["tubes_dets_label"]

        match_logits = match_logits / 0.07

        match_logits = match_logits.masked_fill(dets_mask.unsqueeze(1), -float("inf"))
        match_logits = match_logits.masked_fill(
            n_tube_mask.unsqueeze(-1), -float("inf")
        )
        match_logits_log_softmax = F.log_softmax(match_logits, dim=2)

        loss_match = match_logits_log_softmax.view(-1, match_logits.size(-1))
        loss_match = -loss_match[range(loss_match.size(0)), tubes_dets_label.view(-1)]
        loss_match = loss_match.reshape(*match_logits.shape[:-1])

        loss_match = loss_match[~n_tube_mask].mean()

        return {"loss_match": loss_match}

    def loss_boxes(self, outputs, targets):
        """ outputs consists of:
                - "n_tubes_mask" (N, max_n_tube)
                - "pred_boxes" (N, max_n_tube, 4)
            targets consists of:
                - "boxes" (N, max_n_tube, 4)
                - "cont" (N, max_n_tube)
        """
        n_tube_mask = outputs["n_tubes_mask"].reshape(-1)
        pred_boxes = outputs["pred_boxes"].reshape(-1, 4)
        boxes = targets["bboxes"].reshape(-1, 4)
        labels = targets["cont"].reshape(-1)
        valid_mask = (~n_tube_mask) & labels

        loss_bbox = F.l1_loss(pred_boxes, boxes, reduction="none").sum(-1)
        loss_bbox = loss_bbox[valid_mask].mean()

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(pred_boxes),
                box_ops.box_cxcywh_to_xyxy(boxes),
            )
        )
        loss_giou = loss_giou[valid_mask].mean()

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "match": self.loss_match,
        }

        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_bbox = out_bbox.reshape(*out_bbox.shape[:-1], -1, 4)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        scores = out_logits.sigmoid()
        labels = (scores > 0.5).int()

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


def build(args):
    device = torch.device(args.device)

    pos_embed = PositionalEncoding(d_model=args.hidden_dim, max_len=args.tube_len)
    transformer_ = TrackTrans(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_track_encoder_layers=args.num_track_encoder_layers,
        num_det_encoder_layers=args.num_det_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pred_only=args.pred_only,
    )
    if args.extract_feat:
        feat_extractor = FRCNNFeatureExtractor(args.pretrained_feat_extractor)
    else:
        feat_extractor = None

    model = TT(
        pos_embed, transformer_, feat_dim=args.feat_dim, feat_extractor=feat_extractor
    )

    # losses = ["labels", "boxes"]
    losses = ["boxes"]
    if args.match_loss_coef > 0.0:
        losses.append("match")
    weight_dict = {
        "loss_match": args.match_loss_coef,
        "loss_bce": args.bce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    criterion = SetCriterion(losses, weight_dict)
    criterion.to(device)
    postproessors = {"bbox": PostProcess()}

    return model, criterion, postproessors
