# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

from detectron2.config import configurable
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

__all__ = ["WeakROIHeads"]

logger = logging.getLogger(__name__)


class WeakROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        proposal_matcher: Matcher,
        proposal_matcher_pcl: Matcher,
        proposal_append_gt: bool = True,
        use_pcl_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_matcher_pcl (Matcher): proposal cluster matcher that matches proposals
            and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
            use_pcl_loss (bool): whether to use proposal cluster loss which calculates the loss
            for each cluster instead of each proposal
        """
        super().__init__()
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_matcher_pcl = proposal_matcher_pcl
        self.proposal_append_gt = proposal_append_gt
        self.use_pcl_loss = use_pcl_loss

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
            "use_pcl_loss": cfg.MODEL.ROI_BOX_HEAD.USE_PCL_LOSS,
            # Matcher to assign box proposals to pseudo gt boxes achieved by proposal cluster
            "proposal_matcher_pcl": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS_PCL,
                cfg.MODEL.ROI_HEADS.IOU_LABELS_PCL,
                allow_low_quality_matches=False,
            ),
        }

    @torch.no_grad()
    def get_image_level_gt(self, targets: List[Instances]) -> Tuple[torch.Tensor, torch.Tensor]:
        img_classes = [torch.unique(t.gt_classes, sorted=True) for t in targets]
        img_classes = [gt.to(torch.int64) for gt in img_classes]
        img_classes_oh = torch.cat(
            [
                torch.zeros(
                    (1, self.num_classes), dtype=torch.float, device=img_classes[0].device
                ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
                for gt in img_classes
            ],
            dim=0,
        )
        img_classes_oh = torch.cat(
            (img_classes_oh, img_classes_oh.new_ones((len(targets), 1))), dim=1
        )
        return img_classes, img_classes_oh

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
            gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
            (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
            each sampled proposal. Each sample is labeled as either a category in
            [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        # sampled_idxs = torch.where(matched_labels == 1)[0]
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self,
        proposals: List[Instances],
        targets: List[Instances],
        is_PCL: bool = False,
        suffix: str = "",
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            if is_PCL:
                matched_idxs, matched_labels = self.proposal_matcher_pcl(match_quality_matrix)
            else:
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set gt_classes attribute of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # Set gt_boxes attribute of the sampled proposals:
            sampled_targets = matched_idxs[sampled_idxs]
            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]
            else:
                proposals_per_image.gt_boxes = proposals_per_image.proposal_boxes.clone()

            # Set gt_scores attribute of the sampled proposals:
            if targets_per_image.has("gt_scores"):
                if has_gt:
                    gt_scores = targets_per_image.gt_scores[sampled_targets]
                else:
                    gt_scores = torch.ones_like(sampled_targets, dtype=targets_per_image.gt_scores.dtype)
                proposals_per_image.gt_scores = gt_scores

            # Set gt_clusters attribute of the sampled proposals:
            if self.use_pcl_loss:
                gt_clusters = matched_idxs.clone()
                gt_clusters[(matched_labels == 0) | (matched_labels == -1)] = -1
                gt_clusters = gt_clusters[sampled_idxs]
                proposals_per_image.gt_clusters = gt_clusters

            if has_gt:
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_ig_samples[-1] - num_bg_samples[-1])

            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()
