<p align="center">

  <h1 align="center">HyHOT: Learning Human-Centric Priors for Fine-Grained Human-Object Contact Segmentation
</h1>
  <p align="center">
    <strong>HybridHOT: Learning Human-Centric Priors for Fine-Grained Human-Object Contact Segmentation</strong>
  </p>
  <div align="center">
    <img src="./assets/paper.png" alt="Logo" width="100%">
  </div>

## Environment
Please first install the following environment:
- python 3.11
- pytorch 2.5.1 (cu121)
- torchvision 0.19.1 (cu121)

## Installation
```
pip3 install -r requirements.txt
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
|   |   |   |── ...
|   |   |── HOT-Generated
|   |   |   |── images
|   |   |   |── annotations
|   |   |   |── ...
│   ├── hot_train.odgt
│   ├── hot_test.odgt
│   ├── ...

```

## Segmentation Model

We use the [SAM GitHub](https://github.com/paulguerrero/lang-sam) model to generate human masks and save them in the `./data/HOT/HOT-Annotated(HOT-Generated)/segments_lang_sam` directory.


## Depth Model
[ZoeDepth](https://github.com/isl-org/ZoeDepth) is used to generate depth map. [LaMa](https://hot.is.tue.mpg.de) model in combination with the human mask to reconstruct the occluded object information.
Developers need to install the environment according to the official instructions of [ZoeDepth](https://github.com/isl-org/ZoeDepth) and save the generated depth map to the `./data/HOT/HOT-Annotated(HOT-Generated)/depth` directory.
Please note that in order to keep the original image and the inpainting image at the same perspective, they need to be spliced ​​together and sent to the [ZoeDepth](https://github.com/isl-org/ZoeDepth) model.


## Sapiens2 Encoder
HyHOT uses the [Sapiens2](https://about.meta.com/realitylabs/codecavatars/sapiens) ViT backbone as the image encoder. Place the Sapiens2 source code under `./sapiens2/` so that `sapiens.backbones.standalone.sapiens2` can be imported, and put the pretrained weights under `./sapiens2/sapiens2_host/`.

Supported `MODEL.sapiens_arch` variants (set in `config/hot-sapiens-hyhot.yaml`):

| arch            | embed_dims | num_layers |
|-----------------|------------|------------|
| sapiens2_0.1b   | 768        | 12         |
| sapiens2_0.4b   | 1024       | 24         |
| sapiens2_0.8b   | 1280       | 32         |
| sapiens2_1b     | 1536       | 40         |
| sapiens2_5b     | 2432       | 56         |

The default config loads `sapiens2_0.4b` with pretrained weights at
`./sapiens2/sapiens2_host/sapiens2_0.4b_seg.safetensors`.


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

```bibtex
TODO
```

## Acknowledgement
For the HOT model and dataset proposed by Chen et al., please click [HOT](https://github.com/yixchen/HOT) for details.
These depth maps and masks were generated following the method proposed by Wang et al. [P3HOT](https://arxiv.org/abs/2507.01630) (ICCV 2025) for details.
