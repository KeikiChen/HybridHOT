# System libs
import os
import sys
import time
import random
import argparse
import contextlib
from datetime import datetime
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import wandb
except ImportError:
    wandb = None
# Our libs
from hot.config import cfg
from hot.dataset import TrainDataset
from hot.models import ModelBuilder, SegmentationModule
from hot.utils import AverageMeter, setup_logger
from hot.lib.nn import user_scattered_collate
import warnings
warnings.filterwarnings("ignore")

TRAIN_INFO_PATH = None  # set in __main__ (rank-0 only)


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _to_cuda(obj, device, non_blocking=True):
    # DDP does not scatter inputs (unlike DataParallel); each rank's dataloader
    # returns CPU tensors that must be pushed to its own device.
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_cuda(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_cuda(v, device, non_blocking) for v in obj)
    return obj


def _dump_train_log(dataset_train, cfg, argv):
    train_log_path = os.path.join(cfg.DIR, "train.log")
    with open(train_log_path, "w") as f:
        f.write("===== start time =====\n")
        f.write("{}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        f.write("===== command =====\n")
        f.write("{}\n\n".format(" ".join(argv)))

        f.write("===== model params (cfg) =====\n")
        f.write("{}\n\n".format(cfg))

        f.write("===== human-mask & depth source folders (TrainDataset) =====\n")
        if len(dataset_train.list_sample) > 0:
            sample = dataset_train.list_sample[0]
            image_path = os.path.join(cfg.DATASET.root_dataset, sample['fpath_img'])
            segm_path = os.path.join(cfg.DATASET.root_dataset, sample['fpath_segm'])
            depth_path  = segm_path.replace("/annotations/", "/{}/".format(cfg.DATASET.depth_subdir)).replace(".png", ".npy")
            person_mask = image_path.replace("/images/",     "/{}/".format(cfg.DATASET.person_mask_subdir)).replace(".jpg", ".npy")

            f.write("human_mask_dir: {}\n".format(os.path.dirname(person_mask)))
            f.write("depth_dir: {}\n".format(os.path.dirname(depth_path)))
            f.write("num_samples: {}\n".format(len(dataset_train.list_sample)))
        f.write("===== end =====\n")


def _append_train_end_time(cfg):
    train_log_path = os.path.join(cfg.DIR, "train.log")
    with open(train_log_path, "a") as f:
        f.write("\n===== end time =====\n")
        f.write("{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def train(ddp_module, iterator, optimizers, history, epoch, cfg, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_s_loss = AverageMeter()
    ave_cross_loss = AverageMeter()
    ave_union_loss = AverageMeter()
    ave_g_loss = AverageMeter()
    ave_smap_loss = AverageMeter()
    ave_dn_loss = AverageMeter()
    ave_contact_loss = AverageMeter()
    ave_resid_loss = AverageMeter()

    ddp_module.train(not cfg.TRAIN.fix_bn)

    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        batch_data = next(iterator)
        batch_data = _to_cuda(batch_data, device)
        data_time.update(time.time() - tic)

        cur_iter = i + (epoch - 1 - cfg.TRAIN.start_epoch) * cfg.TRAIN.epoch_iters

        # Mirror DP (train.py): accumulate 6 micro-batches per optimizer step.
        # This keeps the effective batch / step count / weight-decay frequency
        # identical to train.py, so DDP runs are numerically comparable to the
        # DP baseline. `no_sync()` skips the all-reduce on the 5 intermediate
        # backwards in each window for free speedup.
        do_step = ((i + 1) % 6 == 0)
        sync_ctx = ddp_module.no_sync() if not do_step else contextlib.nullcontext()
        with sync_ctx:
            (loss, s_loss, cross_loss, union_loss, g_loss,
             smap_loss, dn_loss, contact_loss, resid_loss, acc) = \
                ddp_module(batch_data)

            loss = loss.mean() / 6
            acc = acc.mean()
            loss.backward()

        if do_step:
            adjust_learning_rate(optimizers, int(cur_iter // 6), cfg)
            for optimizer in optimizers:
                optimizer.step()
            ddp_module.zero_grad()

        loss *= 6  # restore reported loss to full-batch scale (matches train.py)

        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_total_loss.update(loss.data.item())
        ave_s_loss.update(s_loss.mean().data.item())
        ave_cross_loss.update(cross_loss.mean().data.item())
        ave_union_loss.update(union_loss.mean().data.item())
        ave_g_loss.update(g_loss.mean().data.item())
        ave_smap_loss.update(smap_loss.mean().data.item())
        ave_dn_loss.update(dn_loss.mean().data.item())
        ave_contact_loss.update(contact_loss.mean().data.item())
        ave_resid_loss.update(resid_loss.mean().data.item())
        ave_acc.update(acc.data.item() * 100)

        if i % cfg.TRAIN.disp_iter == 0 and is_main():
            msg = (
                'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                'Accuracy: {:4.2f}, Loss: {:.6f}\n'
                ' s_loss: {:.4f}, cross_loss: {:.4f}, union_loss: {:.4f}, '
                'g_loss: {:.4f}, smap_loss: {:.4f}, dn_loss: {:.4f}, '
                'contact_loss: {:.4f}, resid_loss: {:.4f}'
            ).format(
                epoch, i, cfg.TRAIN.epoch_iters,
                batch_time.average(), data_time.average(),
                cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                ave_acc.average(), ave_total_loss.average(),
                ave_s_loss.average(), ave_cross_loss.average(),
                ave_union_loss.average(), ave_g_loss.average(),
                ave_smap_loss.average(), ave_dn_loss.average(),
                ave_contact_loss.average(), ave_resid_loss.average(),
            )
            print(msg)
            with open(TRAIN_INFO_PATH, "a+") as f:
                f.write(msg + "\n")

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    'train/epoch': fractional_epoch,
                    'train/iter': cur_iter,
                    'train/loss': ave_total_loss.average(),
                    'train/loss_step': loss.data.item(),
                    'train/acc': ave_acc.average(),
                    'train/s_loss': ave_s_loss.average(),
                    'train/cross_loss': ave_cross_loss.average(),
                    'train/union_loss': ave_union_loss.average(),
                    'train/g_loss': ave_g_loss.average(),
                    'train/smap_loss': ave_smap_loss.average(),
                    'train/dn_loss': ave_dn_loss.average(),
                    'train/contact_loss': ave_contact_loss.average(),
                    'train/resid_loss': ave_resid_loss.average(),
                    'train/lr_encoder': cfg.TRAIN.running_lr_encoder,
                    'train/lr_decoder': cfg.TRAIN.running_lr_decoder,
                    'train/batch_time': batch_time.average(),
                    'train/data_time': data_time.average(),
                }, step=cur_iter)


def checkpoint(nets, history, cfg, epoch):
    if not is_main():
        return
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    torch.save(history, '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(net_encoder.state_dict(),
               '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(net_decoder.state_dict(),
               '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def _encoder_train_params(net_encoder):
    # CLIP RN50: train only the visual encoder (preserve original P3HOT
    # behavior). Sapiens / other encoders: train all parameters.
    if hasattr(net_encoder, "visual") and not hasattr(net_encoder, "backbone"):
        return net_encoder.visual.parameters()
    return net_encoder.parameters()


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.AdamW(
        _encoder_train_params(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        weight_decay=1e-4)

    optimizer_decoder = torch.optim.AdamW(
        net_decoder.parameters(),
        lr=cfg.TRAIN.lr_decoder,
        weight_decay=1e-4)

    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, local_rank, world_size):
    device = torch.device("cuda", local_rank)

    # ---- Build encoder / decoder via ModelBuilder ----
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
        cfg=cfg,
    )
    net_decoder = ModelBuilder.build_decoder(
        cfg=cfg,
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=False,
    )

    weight = np.ones((cfg.DATASET.num_class), dtype=np.float32)
    weight[0] = 0.02
    weight = torch.tensor(weight).float()
    crit = nn.CrossEntropyLoss(weight=weight)

    segmentation_module = SegmentationModule(
        net_encoder, net_decoder, crit, cfg=cfg)
    segmentation_module.to(device)

    # Per-rank dataset. TrainDataset.__getitem__ self-batches to batch_per_gpu,
    # so we let DataLoader use batch_size=1 + user_scattered_collate (returns
    # a list of one dict, matching SegmentationModule.forward's [dict] shape).
    # Different ranks must see different shuffles to avoid 8x duplicate reads.
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    rank_seed = cfg.TRAIN.seed + local_rank * 9973
    _rng = np.random.RandomState(rank_seed)
    _perm = _rng.permutation(len(dataset_train.list_sample)).tolist()
    dataset_train.list_sample = [dataset_train.list_sample[i] for i in _perm]
    dataset_train.if_shuffled = True  # skip dataset's own first-call shuffle

    loader_kwargs = dict(
        batch_size=1,                       # dataset already batches to bs/gpu
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True,
    )
    if cfg.TRAIN.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader_train = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)

    if is_main():
        print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
        _dump_train_log(dataset_train, cfg, sys.argv)

    iterator_train = iter(loader_train)

    # DDP wrap. broadcast_buffers=False preserves per-rank BN running stats,
    # matching the DP run (sapiens encoder + p3hot/hhd decoder use plain
    # nn.BatchNorm2d, not SyncBN, so each GPU already had independent stats).
    # find_unused_parameters=True is conservative — optional losses can leave
    # some heads unused. Set to False once you confirm all params participate.
    ddp_module = DDP(
        segmentation_module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(ddp_module, iterator_train, optimizers, history, epoch + 1, cfg, device)
        if dist.is_initialized():
            dist.barrier()
        checkpoint(nets, history, cfg, epoch + 1)

    if is_main():
        _append_train_end_time(cfg)
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.10.0'), \
        'PyTorch>=1.10 is required for DDP'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training (DDP, torchrun)"
    )
    parser.add_argument(
        "--cfg",
        default="config/hot-resnet50dilated-c1.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # torchrun-provided env vars
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    logger = setup_logger(distributed_rank=local_rank)
    if local_rank == 0:
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))

    if local_rank == 0 and not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    if local_rank == 0:
        logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
        with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
            f.write("{}".format(cfg))

    TRAIN_INFO_PATH = os.path.join(cfg.DIR, "train_info.txt")
    if local_rank == 0 and os.path.exists(TRAIN_INFO_PATH):
        os.remove(TRAIN_INFO_PATH)

    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    cfg.TRAIN.batch_size = world_size * cfg.TRAIN.batch_size_per_gpu
    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * (cfg.TRAIN.num_epoch - cfg.TRAIN.start_epoch)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed + local_rank)
    np.random.seed(cfg.TRAIN.seed + local_rank)
    torch.manual_seed(cfg.TRAIN.seed + local_rank)

    torch.backends.cudnn.benchmark = True

    if local_rank == 0 and wandb is not None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "HybridHOT"),
            name=os.environ.get("WANDB_RUN_NAME", os.path.basename(cfg.DIR.rstrip("/"))),
            dir=cfg.DIR,
            config={"cfg_file": args.cfg, "world_size": world_size},
        )

    main(cfg, local_rank, world_size)
    dist.destroy_process_group()
