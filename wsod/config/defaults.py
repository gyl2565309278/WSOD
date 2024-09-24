# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_wsod_cfg(cfg):
    """
    Add config for WSOD nets.
    """

    # -----------------------------------------------------------------------------
    # Config definition
    # -----------------------------------------------------------------------------
    _C = cfg
    _C.EPSILON = 1e-12


    # ---------------------------------------------------------------------------- #
    # VGGNETS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.VGGNETS = CN()

    _C.MODEL.VGGNETS.DEPTH = 16
    _C.MODEL.VGGNETS.OUT_FEATURES = ["vgg5"]  # vgg4 for C4 backbone, vgg2..5 for FPN backbone

    # Number of groups to use; 1 ==> VGGNet; > 1 ==> VGGNeXt
    _C.MODEL.VGGNETS.NUM_GROUPS = 1

    # Options: "", "FrozenBN", "GN", "SyncBN", "BN"
    _C.MODEL.VGGNETS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.VGGNETS.WIDTH_PER_GROUP = 64

    # Apply dilation in stage "vgg5"
    _C.MODEL.VGGNETS.VGG5_DILATION = 1

    # Output width of vgg1. Scaling this parameters will scale the width of all 3x3 convs in VGGNet
    # This needs to be set to 64
    _C.MODEL.VGGNETS.VGG1_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on vgg1, vgg2, vgg3, vgg4, vgg5
    _C.MODEL.VGGNETS.DEFORM_ON_PER_STAGE = [False, False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.VGGNETS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.VGGNETS.DEFORM_NUM_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # VGGNETSWS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.VGGNETSWS = CN()

    _C.MODEL.VGGNETSWS.DEPTH = 16
    _C.MODEL.VGGNETSWS.OUT_FEATURES = ["vgg5"]  # vgg4 for C4 backbone, vgg2..5 for FPN backbone

    # Number of groups to use; 1 ==> VGGNet-WS; > 1 ==> VGGNeXt-WS
    _C.MODEL.VGGNETSWS.NUM_GROUPS = 1

    # Options: "", "FrozenBN", "GN", "SyncBN", "BN"
    _C.MODEL.VGGNETSWS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.VGGNETSWS.WIDTH_PER_GROUP = 64

    # Apply dilation in stage "vgg5"
    _C.MODEL.VGGNETSWS.VGG5_DILATION = 1

    # Output width of vgg1. Scaling this parameters will scale the width of all 3x3 convs in VGGNet-WS
    # This needs to be set to 64
    _C.MODEL.VGGNETSWS.VGG1_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on vgg1, vgg2, vgg3, vgg4, vgg5
    _C.MODEL.VGGNETSWS.DEFORM_ON_PER_STAGE = [False, False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.VGGNETSWS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.VGGNETSWS.DEFORM_NUM_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # RESNETSWS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETSWS = CN()

    _C.MODEL.RESNETSWS.DEPTH = 50
    _C.MODEL.RESNETSWS.OUT_FEATURES = ["res5"]

    # Number of groups to use; 1 ==> ResNet-WS; > 1 ==> ResNeXt-WS
    _C.MODEL.RESNETSWS.NUM_GROUPS = 1

    # Options: "", "FrozenBN", "GN", "SyncBN", "BN"
    _C.MODEL.RESNETSWS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.RESNETSWS.WIDTH_PER_GROUP = 64

    # Apply dilation in stage "res5"
    _C.MODEL.RESNETSWS.RES5_DILATION = 1

    # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet-WS
    # For R18 and R34, this needs to be set to 64
    _C.MODEL.RESNETSWS.RES2_OUT_CHANNELS = 256
    _C.MODEL.RESNETSWS.STEM_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    _C.MODEL.RESNETSWS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.RESNETSWS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.RESNETSWS.DEFORM_NUM_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # ROI_HEADS options
    # ---------------------------------------------------------------------------- #
    # Options: "WSDDN", "SRWSDDN"
    _C.MODEL.ROI_HEADS.BASE_NAME = "WSDDN"

    _C.MODEL.ROI_HEADS.IOU_THRESHOLDS_PCL = [0.1, 0.5]
    _C.MODEL.ROI_HEADS.IOU_LABELS_PCL = [-1, 0, 1]

    # ---------------------------------------------------------------------------- #
    # ROI_BOX_HEAD options
    # ---------------------------------------------------------------------------- #
    # Choose whether to use proposal cluster loss
    _C.MODEL.ROI_BOX_HEAD.USE_PCL_LOSS = False

    # ---------------------------------------------------------------------------- #
    # WSOD options
    # ---------------------------------------------------------------------------- #
    _C.WSOD = CN()
    _C.WSOD.VIS_TEST = False

    _C.WSOD.CSC_MAX_ITER = 35000

    _C.WSOD.REFINE_K = 4
    _C.WSOD.REFINE_REG = [False, False, False, False]
    _C.WSOD.BOX_REG = False

    # ---------------------------------------------------------------------------- #
    # PCL options
    # ---------------------------------------------------------------------------- #
    _C.WSOD.PCL = CN()
    # If the remaining verticles is smaller than MIN_REMAIN_COUNT, stop filter pseudo gt
    _C.WSOD.PCL.MIN_REMAIN_COUNT = 5
    # The maximum number of proposal clusters per class
    _C.WSOD.PCL.MAX_NUM_PC = 5

    _C.WSOD.PCL.KMEANS = CN()
    _C.WSOD.PCL.KMEANS.NUM_CLUSTERS = 3
    _C.WSOD.PCL.KMEANS.SEED = 2

    _C.WSOD.PCL.GRAPH = CN()
    _C.WSOD.PCL.GRAPH.IOU_THRESHOLD = 0.4

    # ---------------------------------------------------------------------------- #
    # TEST options
    # ---------------------------------------------------------------------------- #
    _C.TEST.EVAL_TRAIN = True
    # Options: "average", "union"
    _C.TEST.AUG.TYPE = "average"
