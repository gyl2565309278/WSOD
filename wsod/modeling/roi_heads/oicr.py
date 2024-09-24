# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage

from .weak_roi_heads import WeakROIHeads
from .wsddn import WSDDNOutputLayers

__all__ = ["oicr_inference", "OICROutputLayers", "OICRROIHeads"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def oicr_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `oicr_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`WSDDNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`WSDDNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        oicr_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return (
        [x[0] for x in result_per_image],
        [x[1] for x in result_per_image],
        [x[2] for x in result_per_image],
        [x[3] for x in result_per_image],
    )


def oicr_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `oicr_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `oicr_inference`, but for only one image.
    """
    all_boxes = torch.unsqueeze(boxes.clone(), 0)
    all_scores = torch.unsqueeze(scores.clone(), 0)

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0], all_boxes, all_scores


def _log_classification_stats(pred_logits, gt_classes, prefix="oicr"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


class OICROutputLayers(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
        refine_k: int,
        has_reg: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        self.num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.box_dim = len(box2box_transform.weights)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.box2box_transform = box2box_transform
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        if isinstance(loss_weight, float):
            if not has_reg:
                loss_weight = {"loss_cls_r{}".format(refine_k): loss_weight}
            else:
                loss_weight = {
                    "loss_cls_r{}".format(refine_k): loss_weight,
                    "loss_box_reg_r{}".format(refine_k): loss_weight,
                }
        self.loss_weight = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes

        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Please use sigmoid cross entropy loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)
            self.fed_loss_cls_weights = fed_loss_cls_weights

        if has_reg:
            self.bbox_pred = nn.Linear(input_size, self.num_bbox_reg_classes * self.box_dim)
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            nn.init.constant_(self.bbox_pred.bias, 0)
            self.smooth_l1_beta = smooth_l1_beta
            self.box_reg_loss_type = box_reg_loss_type

        self.refine_k = refine_k
        self.has_reg = has_reg

    @classmethod
    def from_config(cls, cfg, input_shape, refine_k, has_reg):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"            : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"         : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"               : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            "use_fed_loss"              : cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            "use_sigmoid_ce"            : cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "get_fed_loss_cls_weights"  : lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER),  # noqa
            "fed_loss_num_classes"      : cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES,
            # fmt: on
            "refine_k"                  : refine_k,
            "has_reg"                   : has_reg,
        }

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        if self.has_reg:
            proposal_deltas = self.bbox_pred(x)
        else:
            proposal_deltas = scores.new_zeros(
                (scores.shape[0], self.num_bbox_reg_classes * self.box_dim),
                requires_grad=False,
            )
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
            to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
            ``gt_classes``, ``gt_scores`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes, prefix="oicr_r{}".format(self.refine_k))

        gt_scores = (
            cat([p.gt_scores for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_clss = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_clss = cross_entropy(scores, gt_classes, reduction="none", ignore_index=-1)
        N = (gt_classes != -1).sum().item()
        loss_cls = torch.sum(loss_clss * gt_scores) / N

        if not self.has_reg:
            losses = {"loss_cls_r{}".format(self.refine_k): loss_cls}
        else:
            losses = {
                "loss_cls_r{}".format(self.refine_k): loss_cls,
                "loss_box_reg_r{}".format(self.refine_k): self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                ),
            }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/fed_loss.py  # noqa
    # with slight modifications
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
    # with slight modifications
    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        N = pred_class_logits.shape[0]
        K = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(N, K + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :K]

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction="none"
        )

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1

        loss = torch.sum(cls_loss * weight, dim=1)
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(
            self,
            predictions_all_stages: List[Tuple[torch.Tensor, torch.Tensor]],
            proposals: List[Instances],
        ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes_all_stages(predictions_all_stages, proposals)
        scores = self.predict_probs_all_stages(predictions_all_stages, proposals)
        image_shapes = [x.image_size for x in proposals]
        return oicr_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)

            predict_boxes = torch.cat([predict_boxes, proposal_boxes], dim=1)
            predict_boxes = predict_boxes.view(N, K + 1, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes_all_stages(
        self, predictions_all_stages: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]
    ):
        """
        Args:
            predictions_all_stages: return values of :meth:`forward()` in all refine stages.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        proposal_deltas_all_stages = [predictions[1] for predictions in predictions_all_stages]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        for proposal_deltas in proposal_deltas_all_stages:
            predict_boxes = self.box2box_transform.apply_deltas(
                proposal_deltas,
                proposal_boxes,
            )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_probs_all_stages(
        self, predictions_all_stages: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]
    ):
        """
        Args:
            predictions_all_stages: return values of :meth:`forward()` in all refine stages.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores_all_stages = [predictions[0] for predictions in predictions_all_stages]
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = sum([scores.sigmoid() for scores in scores_all_stages]) / len(scores_all_stages)
        else:
            probs = sum([F.softmax(scores, dim=-1) for scores in scores_all_stages]) / len(scores_all_stages)
        return probs.split(num_inst_per_image, dim=0)


@ROI_HEADS_REGISTRY.register()
class OICRROIHeads(WeakROIHeads):
    """
    It's OICR's head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
        refine_K: int,
        **kwargs,
    ):
        """
        This interface is for WSDDN.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`WSDDNOutputLayers`.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.add_module("box_predictor_0", box_predictor[0])
        for k in range(1, refine_K + 1):
            self.add_module("box_refinery_{}".format(k), box_predictor[k])

        self.train_on_pred_boxes = train_on_pred_boxes
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.refine_K = refine_K

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features            = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution      = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales          = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio         = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg  = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        refine_K               = cfg.WSOD.REFINE_K
        refine_reg             = cfg.WSOD.REFINE_REG
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        box_predictor = [WSDDNOutputLayers(cfg, box_head.output_shape)]
        assert refine_K == len(refine_reg), "{} != {}".format(refine_K, len(refine_reg))
        for k in range(1, refine_K + 1):
            box_predictor.append(OICROutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1]))

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "refine_K": refine_K,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Union[
        Tuple[List[Instances], Dict[str, torch.Tensor]],
        Tuple[List[Instances], Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
    ]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            img_classes, img_classes_oh = self.get_image_level_gt(targets)
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, img_classes_oh, img_classes)
            return proposals, losses
        else:
            pred_instances, all_boxes, all_scores = self._forward_box(features, proposals)
            return pred_instances, {}, all_boxes, all_scores

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        img_classes_oh: Optional[torch.Tensor] = None,
        img_classes: Optional[List[torch.Tensor]] = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        # Maybe should be removed?
        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        box_features = self.box_head(box_features)

        if self.training:
            with torch.no_grad():
                origin_proposal_boxes = [
                    proposals_per_image.proposal_boxes.clone() for proposals_per_image in proposals
                ]

            predictions = self.box_predictor[0](box_features, proposals)
            losses = self.box_predictor[0].losses(predictions, proposals, img_classes_oh)

            with torch.no_grad():
                boxes = self.box_predictor[0].predict_boxes(predictions, proposals)
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
            for k in range(1, self.refine_K + 1):
                proposals = self._proposal_cluster_learning(
                    boxes, scores, proposals, img_classes, suffix="_r{}".format(k)
                )

                predictions = self.box_predictor[k](box_features)

                losses_k = self.box_predictor[k].losses(predictions, proposals)
                losses.update(losses_k)

                with torch.no_grad():
                    boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                    scores = self.box_predictor[k].predict_probs(predictions, proposals)

                    pred_boxes = self.box_predictor[k].predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            # proposals is modified in-place below, so losses must be computed first.
            if not self.train_on_pred_boxes:
                with torch.no_grad():
                    for proposals_per_image, origin_proposal_boxes_per_image in zip(
                        proposals, origin_proposal_boxes
                    ):
                        proposals_per_image.proposal_boxes = origin_proposal_boxes_per_image
            return losses
        else:
            predictions_all_stages = []
            for k in range(1, self.refine_K + 1):
                predictions = self.box_predictor[k](box_features)
                predictions_all_stages.append(predictions)

            pred_instances, _, all_boxes, all_scores = self.box_predictor[-1].inference(
                predictions_all_stages, proposals
            )
            return pred_instances, all_boxes, all_scores

    @torch.no_grad()
    def _proposal_cluster_learning(
        self,
        boxes: Tuple[torch.Tensor, ...],
        scores: Tuple[torch.Tensor, ...],
        proposals: List[Instances],
        img_classes: List[torch.Tensor],
        suffix: str,
    ) -> List[Instances]:
        scores = [
            scores_per_image.index_select(1, img_classes_per_image)
            for scores_per_image, img_classes_per_image in zip(scores, img_classes)
        ]

        if self.cls_agnostic_bbox_reg:
            boxes = [
                boxes_per_image.unsqueeze_(1).expand(
                    boxes_per_image.size(0), self.num_classes, boxes_per_image.size(2)
                )
                for boxes_per_image in boxes
            ]
        else:
            boxes = [
                boxes_per_image.view(-1, self.num_classes, 4)
                for boxes_per_image in boxes
            ]
        boxes = [
            boxes_per_image.index_select(1, img_classes_per_image)
            for boxes_per_image, img_classes_per_image in zip(boxes, img_classes)
        ]

        pseudo_targets = self._get_cluster_centers(boxes, scores, proposals, img_classes)

        proposal_append_gt = self.proposal_append_gt
        self.proposal_append_gt = False
        proposals = self.label_and_sample_proposals(proposals, pseudo_targets, suffix=suffix)
        self.proposal_append_gt = proposal_append_gt

        return proposals

    @torch.no_grad()
    def _get_cluster_centers(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        proposals: List[Instances],
        img_classes: List[torch.Tensor],
    ) -> List[Instances]:
        gt_scores, gt_idxs = self._get_highest_scoring_proposals(scores)

        gt_boxes = [
            boxes_per_image.index_select(0, gt_idxs_per_image)
            for boxes_per_image, gt_idxs_per_image in zip(boxes, gt_idxs)
        ]
        gt_boxes = [gt_boxes_per_image.view(-1, 4) for gt_boxes_per_image in gt_boxes]
        diags = [
            torch.tensor(
                [
                    i * gt_classes_per_image.numel() + i
                    for i in range(gt_classes_per_image.numel())
                ],
                dtype=torch.int64,
                device=gt_classes_per_image.device,
            )
            for gt_classes_per_image in img_classes
        ]
        gt_boxes = [
            gt_boxes_per_image.index_select(0, diags_per_image)
            for gt_boxes_per_image, diags_per_image in zip(gt_boxes, diags)
        ]
        gt_boxes = [Boxes(gt_boxes_per_image) for gt_boxes_per_image in gt_boxes]

        gt = [
            Instances(
                proposals_per_image.image_size,
                gt_boxes = gt_boxes_per_image,
                gt_classes = gt_classes_per_image,
                gt_scores = gt_scores_per_image,
            )
            for (
                proposals_per_image, gt_boxes_per_image, gt_classes_per_image, gt_scores_per_image
            ) in zip(proposals, gt_boxes, img_classes, gt_scores)
        ]

        return gt

    @torch.no_grad()
    def _get_highest_scoring_proposals(
        self,
        scores: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        gt_scores = []
        gt_idxs = []

        for scores_per_image in scores:
            gt_scores_per_image, gt_idxs_per_image = torch.max(scores_per_image, dim=0)
            gt_scores.append(gt_scores_per_image)
            gt_idxs.append(gt_idxs_per_image)

        return gt_scores, gt_idxs
