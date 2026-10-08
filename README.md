<p align="center">

  <h1 align="center">HybridHOT: Learning Human-Centric Priors for Fine-Grained Human-Object Contact Segmentation</h1>
  <p align="center">
    <strong>Qihui Chen</strong>
    ·
    <strong>Junwen Chen</strong>
    ·
    <strong>Keiji Yanai</strong>
  </p>
  <h2 align="center">ACCV 2026</h2>
  <div align="center">
    <img src="./assets/paper.png" alt="Logo" width="100%">
  </div>
</p>

## Environment
The code is developed and tested under the following configurations.
- Hardware: 8× NVIDIA RTX A6000 (48 GB) for training, Ubuntu 20.04
- Software: Python 3.12, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1

## Installation
**Clone the repository:**
```bash
git clone https://github.com/KeikiChen/HybridHOT.git
cd HybridHOT
```

**Create the environment** with [uv](https://docs.astral.sh/uv/) and install PyTorch first:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

**Get the Sapiens2 source code.** HyHOT imports [Sapiens2](https://github.com/facebookresearch/sapiens2) from `./sapiens2`, so do not run `pip install -e .` there. It requires `torch>=2.7` and would replace the PyTorch installed above.
```bash
git clone https://github.com/facebookresearch/sapiens2.git
git -C sapiens2 checkout 7e5bae88456ac418ff0e58e74106c9fe192055d4
```

**Install the other dependencies:**
```bash
uv pip install -r requirements.txt
```

## Data Preparation
- Data: download the HOT dataset from the [project website](https://hot.is.tue.mpg.de) and unzip to `/path/to/dataset`. Then:
```
cd ./data
mkdir HOT
ln -s /path/to/dataset ./data/HOT
```
The directory structure is as follows:

```
Project/
├── data/
|   |── HOT
|   |   |── HOT-Annotated
|   |   |   |── images
|   |   |   |── annotations
|   |   |   |── segments_lang_sam    # see Segmentation Model
|   |   |   |── depth                # see Depth Model
|   |   |   |── ...
|   |   |── HOT-Generated
|   |   |   |── images
|   |   |   |── annotations
|   |   |   |── segments_lang_sam
|   |   |   |── depth
|   |   |   |── ...
│   ├── hot_train.odgt
│   ├── hot_test.odgt
│   ├── ...

```

## Segmentation Model
Person masks are generated with [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) and saved to `./data/HOT/HOT-Annotated(HOT-Generated)/segments_lang_sam`. Please refer to [P3HOT](https://github.com/YuxiaoWang-AI/P3HOT) for details.

## Depth Model
Depth maps are generated with [ZoeDepth](https://github.com/isl-org/ZoeDepth), using [LaMa](https://github.com/advimman/lama) inpainting to reconstruct the occluded object regions, and saved to `./data/HOT/HOT-Annotated(HOT-Generated)/depth`. Please refer to [P3HOT](https://github.com/YuxiaoWang-AI/P3HOT) for details.

## Sapiens2 Encoder
HyHOT uses the [Sapiens2](https://github.com/facebookresearch/sapiens2) ViT backbone as the image encoder. Download checkpoints from [MODEL_ZOO.md](https://github.com/facebookresearch/sapiens2/blob/main/docs/MODEL_ZOO.md) and put them under `./sapiens2/sapiens2_host/`. The default config uses Sapiens2-0.4B:
```bash
hf download facebook/sapiens2-seg-0.4b sapiens2_0.4b_seg.safetensors --local-dir sapiens2/sapiens2_host
```
The directory structure is as follows:
```
sapiens2/
├── sapiens/
├── ...
└── sapiens2_host/
    └── sapiens2_0.4b_seg.safetensors
```

Supported `MODEL.sapiens_arch` variants (set in `config/hot-sapiens-hyhot.yaml`). When switching to another variant, also set `MODEL.pretrained` to its checkpoint.

| arch            | embed_dims | num_layers |
|-----------------|------------|------------|
| sapiens2_0.1b   | 768        | 12         |
| sapiens2_0.4b   | 1024       | 24         |
| sapiens2_0.8b   | 1280       | 32         |
| sapiens2_1b     | 1536       | 40         |
| sapiens2_5b     | 2432       | 56         |


## Training
Single-node DDP (default: 8 GPUs):
```
sh ./train.sh
```
or launch manually:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
    train.py --cfg config/hot-sapiens-hyhot.yaml
```
To change which GPUs to use, edit `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` accordingly.

You can change the parameters in `config/hot-sapiens-hyhot.yaml` to adjust the network training process.


## Evaluation
If you want to evaluate the effect of the test set on a specific epoch, you can use the following command:
```
sh ./test.sh
```
This loops over all epochs (1..20 by default). Edit the `EPOCH_NUM` variable or the `--epoch` argument in `test.sh` to evaluate a specific epoch.

After evaluating the model, use the following command to view the results. Note that this command will display the results of all epochs. If you evaluate the validation set first and then evaluate the test set of the specified epoch, the results displayed are the results of the validation set except for the specified test epoch.
```
sh ./show_loss.sh
```

## Citation
This paper has been accepted to ACCV 2026. The proceedings and arXiv version are not yet available; the BibTeX below will be updated once they are released.

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026hybridhot,
  author    = {Chen, Qihui and Chen, Junwen and Yanai, Keiji},
  title     = {{HybridHOT}: Learning Human-Centric Priors for Fine-Grained Human--Object Contact Segmentation},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgement
For the HOT model and dataset proposed by Chen et al., please click [HOT](https://github.com/yixchen/HOT) for details.
The depth maps and person masks are generated following Wang et al.: [PIHOT](https://github.com/YuxiaoWang-AI/PIHOT) (AAAI 2025) and [P3HOT](https://github.com/YuxiaoWang-AI/P3HOT) (ICCV 2025).
