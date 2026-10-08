from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/hot-sapiens-hyhot"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/HOT"
_C.DATASET.list_train = "./data/hot_train.odgt"
_C.DATASET.list_val = "./data/hot_val.odgt"
_C.DATASET.list_test = "./data/hot_test.odgt"
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
# Subdirectory names used to derive depth / person-mask paths from image/segm
# paths. Unset -> P3HOT default ("depth" / "segments_lang_sam"); switch to
# "depth_da3" / "human-mask" via yaml to use the alternative source.
_C.DATASET.depth_subdir = "depth"
_C.DATASET.person_mask_subdir = "segments_lang_sam"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
_C.MODEL.with_part = False

# ---- Sapiens encoder (used when arch_encoder == "sapiens_scls_smap") ----
_C.MODEL.sapiens_arch = "sapiens2_0.4b"
_C.MODEL.pretrained = ""

# ---- HybridHOT decoder hyperparameters (used when arch_decoder in {"hhd","hhd_denoising"}) ----
_C.MODEL.decoder_in_channel = 2048
_C.MODEL.decoder_dim = 256
_C.MODEL.decoder_heads = 8
_C.MODEL.decoder_pixel_layers = 2
_C.MODEL.decoder_layers = 3
_C.MODEL.decoder_queries = 18
_C.MODEL.decoder_dropout = 0.1
_C.MODEL.decoder_drop_path = 0.1
_C.MODEL.decoder_ffn_ratio = 4
_C.MODEL.decoder_smap_alpha_init = 0.0
# DN-specific (used when arch_decoder == "hhd_denoising")
_C.MODEL.dn_groups = 5
_C.MODEL.dn_label_noise = 0.5
_C.MODEL.dn_mask_noise = 0.4
_C.MODEL.dn_loss_weight = 1.0

# ---- P3HOT grouphead decoder (used when arch_decoder == "p3hot_grouphead") ----
# Channel indices (1-based, inside the 17 body-part channels) assigned to each
# group head. Remaining indices auto-fall into the "other" group head.
_C.MODEL.grouphead_hand_indices = (4, 5, 6, 7, 8, 9)
_C.MODEL.grouphead_leg_indices = (10, 11, 12, 13, 14, 15, 16, 17)

# ---- PCID decoder hyperparameters (used when arch_decoder == "pcid") ----
# Object-side ring dilation kernel (odd integer); larger => wider ring.
_C.MODEL.pcid_ring_dilate = 5
# Number of attention heads in the Part-Object Interface Attention.
_C.MODEL.pcid_ring_heads = 4
# Softmax temperature when soft-pooling F_pix into part prototypes.
_C.MODEL.pcid_proto_temperature = 1.0

# ---- Feature toggles (read by SegmentationModule) ----
_C.MODEL.use_sim_loss = True
_C.MODEL.use_union_loss = True
_C.MODEL.use_global_loss = True
_C.MODEL.use_smap_loss = False
_C.MODEL.use_denoising = False
# PCID-specific loss toggles (off by default; only PCID configs enable them).
_C.MODEL.use_contact_loss = False
_C.MODEL.use_residual_loss = False

# Ablation toggles for the Sapiens-Scls-Smap encoder. Defaults preserve the
# original behaviour: both auxiliary heads (S_cls / S_map) are built and used.
# Set to False to drop the head entirely — the encoder then emits a neutral
# placeholder (all-ones prior_logits / no S_map key) so downstream decoders
# and loss paths keep working without code changes.
_C.MODEL.use_cls_head = True
_C.MODEL.use_spatial_head = True

# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.sim_weight = 0.01
_C.LOSS.union_weight = 0.01
_C.LOSS.global_weight = 0.01
_C.LOSS.smap_weight = 0.01
_C.LOSS.denoising_weight = 0.1
# PCID-specific weights. Defaults are inert for non-PCID configs because the
# corresponding losses resolve to zero unless the decoder caches `ring_mask`
# and `use_contact_loss` / `use_residual_loss` are turned on.
_C.LOSS.contact_weight = 0.5
_C.LOSS.residual_weight = 0.005

# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
_C.DEBUG = CN()
_C.DEBUG.print_shape = False

# -----------------------------------------------------------------------------
# Eval (extension; baseline P3HOT eval ignores these)
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.save_epoch_metrics = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
# output visualization during validation
_C.TEST.visualize = True
