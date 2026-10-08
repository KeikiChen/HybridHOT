"""Inference + visualization on the test split (cfg.DATASET.list_test).

Loads encoder_epoch_<N>.pth / decoder_epoch_<N>.pth from cfg.DIR and saves
`[image | prediction overlay]` images to `<cfg.DIR>/result_epoch_<N>/`.

Usage (from HybridHOT_release/):

    python eval_inference.py \\
        --cfg config/hot-sapiens-hyhot.yaml \\
        --gpu 0 --epoch 20
"""

# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
# Our libs
from hot.config import cfg
from hot.dataset import TestDataset
from hot.models import ModelBuilder, SegmentationModule
from hot.utils import colorEncode, setup_logger
from hot.lib.nn import user_scattered_collate, async_copy_to
from hot.lib.utils import as_numpy


with open('data/colors.npy', 'rb') as f:
    colors = np.load(f)


def visualize_result(data, pred, dir_result):
    """Byte-for-byte identical to HOT/eval_inference.py::visualize_result and
    PIHOT/eval_inference.py::visualize_result: horizontal concat of the raw
    image and its `Image.blend(img, colorEncode(pred), 0.7)` overlay.
    """
    (img, _, info) = data

    img_name = info.split('/')[-1]
    img_im = Image.fromarray(img)

    # prediction
    pred_color = colorEncode(pred, colors)
    pred_color_im = Image.fromarray(pred_color)
    comp_pred = Image.blend(img_im, pred_color_im, 0.7)
    comp_pred = np.array(comp_pred)

    im_vis = np.concatenate((img, comp_pred), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name))


def evaluate(segmentation_module, loader, cfg, gpu, epoch):
    print('evaluating epoch:', epoch)

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for idx, batch_data in enumerate(loader):
        bd = batch_data[0]
        info = bd['info']
        img_ori = bd['img_ori']

        # Hyhot dataset returns seg_label at H/4 resolution; use that as the
        # decoder's target output resolution (segSize). This mirrors the
        # convention in eval_metric_epoch.py so downstream vis applies at the
        # "model-native" grid rather than upsampled to the raw image.
        seg_label = as_numpy(bd['seg_label'])
        segSize = (seg_label.shape[0], seg_label.shape[1])

        # Read the RAW image from disk so the visualization matches HOT /
        # PIHOT / P3HOT resolution (raw ~480x640) instead of TestDataset's
        # 224 downscale. Also drives the bilinear upsample target below.
        img_raw = np.array(Image.open(
            os.path.join(cfg.DATASET.root_dataset, info)).convert('RGB'))
        raw_hw = img_raw.shape[:2]

        torch.cuda.synchronize()
        with torch.no_grad():
            feed_dict = bd.copy()
            feed_dict['img_data']    = bd['img_data'].unsqueeze(0)
            feed_dict['depth_label'] = bd['depth_label'].unsqueeze(0)
            feed_dict['person_mask'] = bd['person_mask'].unsqueeze(0)
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, gpu)

            # SegmentationModule returns (pred, final_mask) in eval mode.
            # `pred` is softmax probabilities (use_softmax=True at build).
            scores, _ = segmentation_module(feed_dict, segSize=segSize)

            # Match PIHOT / P3HOT: bilinear-upsample the softmax score map to
            # raw image resolution BEFORE argmax so the class boundary is
            # decided on a continuous probability field — smooth edges at
            # true image resolution instead of nearest-upsample step-ladder.
            scores = nn.functional.interpolate(
                scores, size=raw_hw, mode='bilinear', align_corners=False)
            _, pred = torch.max(scores, dim=1)
            pred_out = as_numpy(pred.squeeze(0).cpu()).astype(np.uint8)

        torch.cuda.synchronize()

        if cfg.TEST.visualize:
            visualize_result(
                (img_raw, None, info),
                pred_out,
                os.path.join(cfg.DIR, 'result_epoch_' + epoch),
            )

        pbar.update(1)


def main(cfg, gpu, epoch):
    torch.cuda.set_device(gpu)

    # Network builders — identical dispatch to eval_metric_epoch.py so every
    # encoder/decoder pair registered in ModelBuilder just works.
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
        use_softmax=True,
    )
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(
        net_encoder, net_decoder, crit, cfg=cfg,
    )
    segmentation_module.cuda(gpu)

    # Dataset and loader — full test set, native order (matches HOT/PIHOT).
    dataset_test = TestDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_test,
        cfg.DATASET,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True,
    )

    evaluate(segmentation_module, loader_test, cfg, gpu, epoch)
    print('Inference done.')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="hyhot: encoder+decoder inference + visualization on TestDataset"
    )
    parser.add_argument(
        "--cfg",
        default="config/hot-sapiens-hyhot.yaml",
        metavar="FILE",
        help="path to a config/*.yaml",
        type=str,
    )
    parser.add_argument("--gpu",   default=0, type=int, help="gpu to use")
    parser.add_argument("--epoch", default="20", type=str,
                        help="checkpoint epoch: loads encoder_epoch_<N>.pth and decoder_epoch_<N>.pth from cfg.DIR")
    parser.add_argument(
        "opts",
        help="cfg overrides",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights (matches eval_metric_epoch.py convention).
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_epoch_' + args.epoch + '.pth')
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_epoch_' + args.epoch + '.pth')
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), \
        "checkpoint does not exist: {} or {}".format(
            cfg.MODEL.weights_encoder, cfg.MODEL.weights_decoder)
    result_dir = os.path.join(cfg.DIR, 'result_epoch_' + args.epoch)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    main(cfg, args.gpu, args.epoch)
